const express = require("express");
const bodyParser = require("body-parser");
const cors = require("cors");
const sql = require("mssql");
const bcrypt = require("bcryptjs");
const sharp = require('sharp');
const { spawn } = require('child_process');
const path = require('path');

const app = express();
app.use(bodyParser.json({ limit: '50mb' }));  // Increased limit for base64 images
app.use(cors());

// Database configuration
const sqlConfig = {
    user: 'henry',
    password: 'henry123',
    server: 'CASPER\\SQLEXPRESS',
    database: 'PAL-AI',
    options: {
        trustServerCertificate: true,
        enableArithAbort: true,
        encrypt: false
    }
};

async function processBase64Image(base64String) {
    try {
        // Remove data:image/jpeg;base64, if present
        const base64Data = base64String.replace(/^data:image\/\w+;base64,/, "");
        const imageBuffer = Buffer.from(base64Data, 'base64');
        
        // Process with sharp
        const processedImage = await sharp(imageBuffer)
            .toFormat('png')
            .toBuffer();
        
        return { image: processedImage, error: null };
    } catch (error) {
        console.error('Image processing error:', error);
        return { image: null, error: error.message };
    }
}

function runYOLOPrediction(imageBuffer) {
    return new Promise((resolve, reject) => {
        const pythonProcess = spawn('python', [
            path.join(__dirname, 'yolo_predict.py')
        ]);

        let stdoutData = '';
        let stderrData = '';

        pythonProcess.stdout.on('data', (data) => {
            stdoutData += data.toString();
        });

        pythonProcess.stderr.on('data', (data) => {
            stderrData += data.toString();
        });

        pythonProcess.stdin.write(imageBuffer);
        pythonProcess.stdin.end();

        pythonProcess.on('close', (code) => {
            // Check for stderr output first (error case)
            if (stderrData) {
                try {
                    const errorObj = JSON.parse(stderrData);
                    reject(new Error(`Python process error: ${JSON.stringify(errorObj)}`));
                    return;
                } catch (parseErr) {
                    reject(new Error(`Python process error (unparseable): ${stderrData}`));
                    return;
                }
            }

            // If no stderr, process stdout
            if (code !== 0) {
                reject(new Error(`Python process exited with code ${code}`));
                return;
            }

            try {
                // Directly parse the stdout
                const predictions = JSON.parse(stdoutData);
                resolve(predictions);
            } catch (e) {
                reject(new Error(`Failed to parse prediction results: ${e.message}. Raw output: ${stdoutData}`));
            }
        });

        pythonProcess.on('error', (err) => {
            reject(new Error(`Python process failed: ${err.message}`));
        });
    });
}

// Scan endpoint
app.post("/scan", async (req, res) => {
    const pool = new sql.ConnectionPool(sqlConfig);
    
    try {
        // Validate incoming data
        const {user_profile_id, disease_prediction, disease_prediction_score } = req.body;
        
        if (!user_profile_id || !disease_prediction || !disease_prediction_score) {
            return res.status(400).json({ message: "Missing required fields" });
        }

        // Connect to database
        await pool.connect();
        
        // Begin transaction
        const transaction = new sql.Transaction(pool);
        await transaction.begin();

        try {
            // Insert into rice_leaf_scan and get the generated ID
            const leafScanResult = await pool.request()
                .input('user_profile_id', sql.VarChar, user_profile_id)
                .input('disease_prediction', sql.VarChar, disease_prediction)
                .input('disease_prediction_score', sql.Float, disease_prediction_score)
                .query(`
                    INSERT INTO rice_leaf_scan (
                        user_profile_id,
                        disease_prediction,
                        disease_prediction_score
                    ) 
                    VALUES (
                        @user_profile_id,
                        @disease_prediction,
                        @disease_prediction_score
                    );
                    SELECT SCOPE_IDENTITY() as rice_leaf_scan_id;
                `);

            const rice_leaf_scan_id = leafScanResult.recordset[0].rice_leaf_scan_id;

            // Insert into scan_history using the obtained rice_leaf_scan_id
            await pool.request()
                .input('rice_leaf_scan_id', sql.Int, rice_leaf_scan_id)
                .query(`
                    INSERT INTO scan_history (
                        rice_leaf_scan_id
                    ) VALUES (
                        @rice_leaf_scan_id 
                    )
                `);

            // Commit transaction
            await transaction.commit();
            res.status(201).json({ 
                message: "Scan data saved successfully",
                rice_leaf_scan_id: rice_leaf_scan_id
            });

        } catch (error) {
            // Rollback transaction if error occurs
            await transaction.rollback();
            throw error;
        }

    } catch (err) {
        console.error('Detailed error:', err);
        res.status(500).json({ 
            message: "Server error during scan data saving",
            error: err.message
        });
    } finally {
        pool.close();
    }
});

//predict endpoint
app.post("/predict", async (req, res) => {
    try {
        const data = req.body;
        
        if (!data || !data.image) {
            return res.status(400).json({
                error: 'No image data provided',
                message: 'Request must include image field with base64 image data'
            });
        }

        // Process the base64 image
        const { image, error } = await processBase64Image(data.image);
        if (error) {
            return res.status(400).json({
                error: 'Failed to process image',
                message: error
            });
        }

        // Add validation for processed image
        if (!image || !Buffer.isBuffer(image)) {
            return res.status(400).json({
                error: 'Invalid image data',
                message: 'Failed to process image into valid buffer'
            });
        }

        console.log('Running YOLO prediction...'); // Debug log
        const predictions = await runYOLOPrediction(image);
        console.log('Predictions received:', predictions); // Debug log

        // Return the predictions
        res.json({
            status: 'success',
            predictions: predictions
        });

    } catch (err) {
        console.error('Prediction error:', err);
        res.status(500).json({
            error: 'Server error',
            message: err.message,
            details: err.stack
        });
    }
});

// Signup endpoint
app.post("/signup", async (req, res) => {
    console.log("Received body:", req.body);  

    const form = req.body;
    const { 
        username, 
        email, 
        password, 
        firstname, 
        lastname, 
        age, 
        gender, 
        mobilenumber 
    } = form;

    try {
        await sql.connect(sqlConfig);

        // Check username or email if already exists
        const existingUser = await sql.query`
            SELECT * FROM users 
            WHERE username = ${username} OR email = ${email}`;
        
        if (existingUser.recordset.length > 0) {
            return res.status(400).json({ message: "Username or email already exists" });
        }

        // Hash password
        const salt = await bcrypt.genSalt(10);
        const hashedPassword = await bcrypt.hash(password, salt);

        // Start transaction
        const transaction = new sql.Transaction();
        await transaction.begin();

        try {
            // Insert into users 
            const userResult = await transaction.request()
                .input('username', sql.NVarChar, username)
                .input('email', sql.NVarChar, email)
                .input('password', sql.NVarChar, hashedPassword)
                .query('INSERT INTO users (username, email, password) VALUES (@username, @email, @password); SELECT SCOPE_IDENTITY() AS user_id;');
            
            const userId = userResult.recordset[0].user_id;
            const parsedAge = form.age ? parseInt(form.age, 10) : null;
            
            // Insert into user_profile table
            await transaction.request()
                .input('user_id', sql.Int, userId)
                .input('firstname', sql.NVarChar, firstname)
                .input('lastname', sql.NVarChar, lastname)
                .input('age', sql.Int, parsedAge)
                .input('gender', sql.NVarChar, gender)
                .input('mobile_number', sql.NVarChar, mobilenumber)
                .query('INSERT INTO user_profile (user_id, firstname, lastname, age, gender, mobile_number) VALUES (@user_id, @firstname, @lastname, @age, @gender, @mobile_number)');

            await transaction.commit();

            res.status(201).json({ message: "User registered successfully", userId });
        } catch (insertError) {
            await transaction.rollback();
            throw insertError;
        }
    } catch (err) {
        console.error(err);
        res.status(500).json({ message: "Server error during registration" });
    } finally {
        await sql.close();
    }
});

// Login endpoint
app.post("/login", async (req, res) => {
    const { username, password } = req.body;
    try {
        await sql.connect(sqlConfig);
        const result = await sql.query`
            SELECT * FROM users 
            WHERE username = ${username}`;
        
        if (result.recordset.length === 0) {
            return res.status(400).json({ message: "User not found" });
        }
        
        const user = result.recordset[0];
        const isMatch = await bcrypt.compare(password, user.password);
        
        if (!isMatch) {
            return res.status(400).json({ message: "Invalid credentials" });
        }
        
        res.json({ 
            message: "Login successful", 
            user: { 
                id: user.user_id, 
                username: user.username 
            } 
        });
    } catch (err) {
        console.error(err);
        res.status(500).json({ message: "Server error" });
    } finally {
        await sql.close();
    }
});

// Home endpoint
app.get("/", (req, res) => {
    res.json({
        status: "online",
        message: "API is running. Send POST requests to /predict"
    });
});


const PORT = 5000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});