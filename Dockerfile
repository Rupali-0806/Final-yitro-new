# Frontend Build Stage
FROM node:18-alpine AS frontend-builder

WORKDIR /app
COPY package*.json ./
RUN npm install

# Copy frontend source and shared files
COPY client/ ./client/
COPY shared/ ./shared/
COPY tailwind.config.ts ./
COPY postcss.config.js ./
COPY components.json ./
COPY vite.config.ts ./
COPY tsconfig.json ./
COPY index.html ./

# Build frontend
RUN npm run build:client

# Backend Build Stage
FROM node:18-alpine AS backend-builder

WORKDIR /app
COPY package*.json ./
RUN npm install

# Copy backend source and shared files
COPY server/ ./server/
COPY shared/ ./shared/
COPY prisma/ ./prisma/
COPY vite.config.server.ts ./
COPY tsconfig.json ./

# Build backend
RUN npm run build:server

# Production Stage
FROM node:18-alpine AS production

WORKDIR /app

# Copy package files and install production dependencies
COPY package*.json ./
RUN npm ci --only=production

# Copy built frontend
COPY --from=frontend-builder /app/dist/spa ./dist/spa

# Copy built backend
COPY --from=backend-builder /app/dist/server ./dist/server

# Copy other necessary files
COPY start.js ./
COPY prisma/ ./prisma/

# Set environment variables
ENV NODE_ENV=production
ENV PORT=3000

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:3000/api/ping || exit 1

# Start the application
CMD ["npm", "run", "start:production"]
