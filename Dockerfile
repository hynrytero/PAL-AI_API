# Use official Node.js runtime
FROM node:18

# Set working directory in container
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy entire project
COPY . .

# Expose port
EXPOSE 5000

# Start the application
CMD ["node", "server.js"]