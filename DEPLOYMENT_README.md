# 🚀 Yitro CRM - Complete Deployment Guide (Docker-Free)

This guide covers deploying your Yitro CRM application to production without Docker. The app is a full-stack React/TypeScript application with a Node.js backend and PostgreSQL database.

## 📋 Prerequisites

- Node.js 20+ installed locally
- Git repository access
- Database provider account (Neon, Supabase, or PlanetScale recommended)
- Deployment platform account (Netlify, Vercel, Railway, etc.)

## 🏗️ Application Architecture

- **Frontend**: React + Vite + TypeScript + Tailwind CSS
- **Backend**: Express.js + Prisma ORM
- **Database**: PostgreSQL (Neon Serverless recommended)
- **Authentication**: JWT-based with bcrypt
- **Email**: Nodemailer integration

## 🗄️ Database Setup (Required First)

### Option 1: Neon Database (Recommended)

```bash
# 1. Sign up at https://neon.tech
# 2. Create a new project
# 3. Copy the connection string
# Format: postgresql://username:password@host/database?sslmode=require
```

### Option 2: Supabase

```bash
# 1. Sign up at https://supabase.com
# 2. Create new project
# 3. Go to Settings > Database
# 4. Copy connection string
```

### Option 3: PlanetScale

```bash
# 1. Sign up at https://planetscale.com
# 2. Create database
# 3. Create branch (main)
# 4. Get connection string
```

## 🌐 Deployment Options

## 🔥 Option 1: Netlify (Easiest - Already Configured)

### Automatic Deployment

1. **Push to Git**:

   ```bash
   git add .
   git commit -m "Deploy to production"
   git push origin main
   ```

2. **Connect to Netlify**:

   - Go to [netlify.com](https://netlify.com)
   - Click "Add new site" → "Import an existing project"
   - Connect your Git repository
   - Netlify will auto-detect the configuration from `netlify.toml`

3. **Set Environment Variables** in Netlify dashboard:
   ```env
   DATABASE_URL=your_neon_connection_string
   JWT_SECRET=your_super_secret_jwt_key_min_32_chars
   EMAIL_HOST=smtp.gmail.com
   EMAIL_PORT=587
   EMAIL_USER=your_email@gmail.com
   EMAIL_PASS=your_app_password
   NODE_ENV=production
   ```

### Manual Deployment

```bash
# Install Netlify CLI
npm install -g netlify-cli

# Login to Netlify
netlify login

# Deploy
netlify deploy --prod
```

---

## ⚡ Option 2: Vercel

### Automatic Deployment

1. **Push to Git** (same as above)

2. **Connect to Vercel**:

   - Go to [vercel.com](https://vercel.com)
   - Click "New Project"
   - Import your repository
   - Configure build settings:
     - **Build Command**: `npm run build`
     - **Output Directory**: `dist/spa`
     - **Install Command**: `npm install && npx prisma generate`

3. **Set Environment Variables** (same as Netlify)

### Manual Deployment

```bash
# Install Vercel CLI
npm install -g vercel

# Login and deploy
vercel login
vercel --prod
```

---

## 🚂 Option 3: Railway

1. **Connect Repository**:

   - Sign up at [railway.app](https://railway.app)
   - Create new project from GitHub repo

2. **Configure Build**:

   - **Build Command**: `npm run build`
   - **Start Command**: `npm run start:production`

3. **Add PostgreSQL Database**:

   - Add PostgreSQL service to your project
   - Railway will provide `DATABASE_URL` automatically

4. **Set Environment Variables** (same as above, except DATABASE_URL is auto-provided)

---

## 🎯 Option 4: Render

1. **Create Web Service**:

   - Go to [render.com](https://render.com)
   - Create new "Web Service"
   - Connect your repository

2. **Configure Build**:

   - **Build Command**: `npm install && npx prisma generate && npm run build`
   - **Start Command**: `npm run start:production`

3. **Add PostgreSQL Database**:
   - Create PostgreSQL database service
   - Copy connection string to environment variables

---

## 🔧 Manual Server Deployment (VPS/Cloud Instance)

### Prerequisites

- Ubuntu/Debian server with root access
- Domain name pointed to server IP

### Setup Process

```bash
# 1. Update system
sudo apt update && sudo apt upgrade -y

# 2. Install Node.js 20
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# 3. Install PM2 for process management
sudo npm install -g pm2

# 4. Clone repository
git clone your-repository-url
cd your-project-name

# 5. Install dependencies
npm install

# 6. Generate Prisma client
npx prisma generate

# 7. Set up environment variables
cp .env.example .env
# Edit .env with your production values

# 8. Run database migrations
npx prisma migrate deploy

# 9. Build application
npm run build

# 10. Start with PM2
pm2 start ecosystem.config.js --env production
pm2 save
pm2 startup
```

### Create PM2 Configuration

Create `ecosystem.config.js`:

```javascript
module.exports = {
  apps: [
    {
      name: "yitro-crm",
      script: "start.js",
      instances: "max",
      exec_mode: "cluster",
      env_production: {
        NODE_ENV: "production",
        PORT: 3000,
      },
    },
  ],
};
```

### Nginx Configuration

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }
}
```

---

## 🔐 Environment Variables Reference

Create these environment variables in your deployment platform:

```env
# Database (Required)
DATABASE_URL=postgresql://username:password@host:port/database?sslmode=require

# Authentication (Required)
JWT_SECRET=your_super_secret_jwt_key_minimum_32_characters_long

# Email Configuration (Optional but recommended)
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USER=your_email@gmail.com
EMAIL_PASS=your_app_specific_password

# Application
NODE_ENV=production
PORT=3000
```

### Getting Gmail App Password

1. Enable 2-factor authentication on Gmail
2. Go to Google Account settings
3. Security → App passwords
4. Generate password for "Mail"
5. Use this password in `EMAIL_PASS`

---

## 🧪 Pre-Deployment Testing

### 1. Test Build Locally

```bash
# Install dependencies
npm install

# Generate Prisma client
npx prisma generate

# Test build
npm run build

# Test production start
npm run start:production
```

### 2. Test Database Connection

```bash
# Test database migration
npx prisma migrate deploy

# Verify connection
npm run test-config
```

### 3. Environment Variables Check

```bash
# Verify all required env vars are set
node -e "
const required = ['DATABASE_URL', 'JWT_SECRET'];
const missing = required.filter(key => !process.env[key]);
if (missing.length) {
  console.error('Missing required environment variables:', missing);
  process.exit(1);
}
console.log('✅ All required environment variables are set');
"
```

---

## 🚨 Troubleshooting

### Common Issues

#### 1. Database Connection Errors

```bash
# Verify DATABASE_URL format
echo $DATABASE_URL

# Test connection
npx prisma db pull
```

#### 2. Build Failures

```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install

# Check for TypeScript errors
npm run typecheck
```

#### 3. Environment Variable Issues

```bash
# Verify env vars are loaded
node -e "console.log(process.env.DATABASE_URL ? '✅ DATABASE_URL set' : '❌ DATABASE_URL missing')"
```

#### 4. Production Login Issues

The app includes test accounts for development:

- **Admin**: admin@yitro.com / admin123
- **User**: user@yitro.com / user123

For production, either:

- Change these credentials in the database
- Create new accounts via the registration flow
- Set up proper user management

---

## 📊 Performance Optimization

### 1. Enable Gzip Compression

Most platforms enable this automatically, but verify in your platform settings.

### 2. Set Up CDN

- Netlify: Automatic global CDN
- Vercel: Automatic Edge Network
- Other platforms: Consider Cloudflare

### 3. Database Optimization

```sql
-- Add indexes for common queries
CREATE INDEX idx_accounts_user_id ON accounts(user_id);
CREATE INDEX idx_activities_account_id ON activities(account_id);
```

---

## 🔄 CI/CD Setup (Optional)

### GitHub Actions Example

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: "20"
      - run: npm install
      - run: npx prisma generate
      - run: npm run build
      - run: npm test
      # Add deployment step for your chosen platform
```

---

## 📞 Support

If you encounter issues:

1. **Check logs** on your deployment platform
2. **Verify environment variables** are set correctly
3. **Test database connection** using provided commands
4. **Review build output** for error messages

For platform-specific issues:

- **Netlify**: Check function logs in dashboard
- **Vercel**: Review build and runtime logs
- **Railway**: Monitor deployment logs
- **Manual deployment**: Check PM2 logs with `pm2 logs`

---

## 🎉 Success!

Once deployed, your Yitro CRM will be available at your chosen platform's URL. The application includes:

- ✅ User authentication and authorization
- ✅ CRM dashboard with metrics
- ✅ Account and contact management
- ✅ Activity tracking
- ✅ Responsive design
- ✅ Email notifications
- ✅ Admin panel

**Next Steps**:

1. Set up your production user accounts
2. Configure email settings for notifications
3. Customize branding and company information
4. Set up monitoring and analytics
5. Configure backup strategies for your database

---

_This deployment guide covers all major platforms and scenarios. Choose the option that best fits your needs and technical expertise._
