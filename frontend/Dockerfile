FROM node:20-alpine

WORKDIR /app

# Install dependencies first (leverages Docker layer cache)
COPY package*.json ./
RUN npm install

# Copy the rest of the frontend source
COPY . .

# Vite dev server port
EXPOSE 5173

# Run Vite dev server and listen on all interfaces
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]
