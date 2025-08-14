# Deployment Guide

This is a full-stack CRM application with clearly separated frontend and backend components.

## Project Structure

```
├── client/           # React Frontend (Vite + TypeScript)
├── server/           # Express Backend API
├── shared/           # Shared types and utilities
├── prisma/           # Database schema and migrations
├── Dockerfile        # Production container
├── Dockerfile.dev    # Development container
└── docker-compose.yml # Multi-service setup
```

## Quick Start

### Option 1: Docker Compose 

```bash
# Development environment
docker-compose up frontend backend db

# Production build test
docker-compose --profile production up app db
```

### Option 2: Manual Setup

```bash
# Install dependencies
npm install

# Development
npm run dev        # Starts both frontend and backend

# Production
npm run build      # Build both frontend and backend
npm run start      # Start production server
```

## Environment Variables

Create a `.env` file:

```env
DATABASE_URL=postgresql://username:password@host:port/database
JWT_SECRET=your-jwt-secret
EMAIL_USER=your-email@domain.com
EMAIL_PASS=your-email-password
```

## Frontend (Port 8080)

- **Framework**: React 18 + TypeScript
- **Build Tool**: Vite
- **Styling**: Tailwind CSS + Radix UI
- **State Management**: React Query + Context
- **Location**: `./client/`

## Backend (Port 3001/3000)

- **Framework**: Express.js + TypeScript
- **Database**: PostgreSQL with Prisma ORM
- **Authentication**: JWT + bcrypt
- **Email**: Nodemailer
- **Location**: `./server/`

## Deployment Options

### 1. Single Container (Production)

```bash
docker build -t crm-app .
docker run -p 3000:3000 -e DATABASE_URL="your-db-url" crm-app
```

### 2. Platform-as-a-Service

- **Heroku**: Use `Dockerfile` with proper buildpacks
- **Railway**: Connect Git repo, auto-deploys
- **DigitalOcean App Platform**: Use docker-compose.yml
- **AWS ECS/Fargate**: Use Dockerfile with task definitions

### 3. Static Frontend + Serverless Backend

- **Frontend**: Deploy `dist/spa` to Netlify/Vercel/Cloudflare
- **Backend**: Deploy as serverless functions (see `netlify/functions/`)

### 4. Kubernetes

```yaml
# k8s deployment example
apiVersion: apps/v1
kind: Deployment
metadata:
  name: crm-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: crm-app
  template:
    metadata:
      labels:
        app: crm-app
    spec:
      containers:
        - name: crm-app
          image: your-registry/crm-app:latest
          ports:
            - containerPort: 3000
          env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: db-secret
                  key: url
```

## Build Commands

```bash
# Frontend only
npm run build:client

# Backend only
npm run build:server

# Both (production)
npm run build

# Test production build
npm run start:production
```

## Development Scripts

```bash
npm run dev          # Full-stack development
npm run test         # Run tests
npm run typecheck    # TypeScript checking
npm run format.fix   # Code formatting
```

## Health Checks

- **API Health**: `GET /api/ping`
- **Frontend**: Check port 8080 accessibility
- **Database**: Prisma connection status

## Troubleshooting

1. **Port conflicts**: Change ports in docker-compose.yml
2. **Database issues**: Check DATABASE_URL and run `npx prisma migrate dev`
3. **Build failures**: Ensure all dependencies in package.json
4. **CORS errors**: Configure backend CORS settings for your domain

For hosting support, contact your platform provider with this deployment guide.
