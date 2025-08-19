#!/bin/bash

# Yitro CRM VPS Setup Script
# Run this on your Ubuntu/Debian VPS server

set -e

echo "🚀 Setting up Yitro CRM on VPS..."
echo "=================================="

# Update system
echo "📦 Updating system packages..."
apt update && apt upgrade -y

# Install essential packages
echo "🔧 Installing essential packages..."
apt install -y curl wget git build-essential software-properties-common

# Install Node.js 20.x
echo "📱 Installing Node.js 20.x..."
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt install -y nodejs

# Verify Node.js installation
node_version=$(node --version)
npm_version=$(npm --version)
echo "✅ Node.js: $node_version"
echo "✅ NPM: $npm_version"

# Install PostgreSQL
echo "🗄️  Installing PostgreSQL..."
apt install -y postgresql postgresql-contrib

# Start and enable PostgreSQL
systemctl start postgresql
systemctl enable postgresql

# Install Nginx
echo "🌐 Installing Nginx..."
apt install -y nginx

# Start and enable Nginx
systemctl start nginx
systemctl enable nginx

# Install PM2 globally
echo "⚡ Installing PM2 process manager..."
npm install -g pm2

# Install Certbot for SSL
echo "🔒 Installing Certbot for SSL..."
apt install -y certbot python3-certbot-nginx

# Create application user
echo "👤 Creating application user..."
useradd -m -s /bin/bash yitro || echo "User 'yitro' already exists"

# Create application directory
echo "📁 Setting up application directory..."
mkdir -p /opt/yitro-crm
chown yitro:yitro /opt/yitro-crm

# Create logs directory
mkdir -p /var/log/yitro-crm
chown yitro:yitro /var/log/yitro-crm

# Setup PostgreSQL database and user
echo "🗄️  Setting up PostgreSQL database..."
sudo -u postgres psql << EOF
CREATE USER yitro_db_user WITH PASSWORD 'YitroSecure123!';
CREATE DATABASE yitro_crm_prod OWNER yitro_db_user;
GRANT ALL PRIVILEGES ON DATABASE yitro_crm_prod TO yitro_db_user;
\\q
EOF

echo "✅ PostgreSQL database 'yitro_crm_prod' created"

# Generate random JWT secret
JWT_SECRET=$(openssl rand -hex 64)

# Create environment file
echo "📝 Creating environment configuration..."
cat > /opt/yitro-crm/.env << EOF
# Yitro CRM Production Environment
NODE_ENV=production
PORT=3000

# Database Configuration
DATABASE_URL=postgresql://yitro_db_user:YitroSecure123!@localhost:5432/yitro_crm_prod

# Security Configuration
STACK_SECRET_SERVER_KEY=$JWT_SECRET

# Application Configuration
FRONTEND_URL=https://dealhub.yitrobc.net

# SMTP Configuration (Optional - update with your details)
SMTP_SERVICE=gmail
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
EOF

chown yitro:yitro /opt/yitro-crm/.env
chmod 600 /opt/yitro-crm/.env

echo "🔐 JWT Secret generated: $JWT_SECRET"
echo "📧 Remember to update SMTP settings in /opt/yitro-crm/.env"

# Configure firewall
echo "🔥 Configuring firewall..."
ufw allow ssh
ufw allow 'Nginx Full'
ufw --force enable

echo ""
echo "🎉 VPS Setup Complete!"
echo "======================"
echo ""
echo "✅ Node.js $node_version installed"
echo "✅ PostgreSQL database ready"
echo "✅ Nginx web server running"
echo "✅ PM2 process manager installed"
echo "✅ Firewall configured"
echo "✅ Application user 'yitro' created"
echo "✅ Environment variables configured"
echo ""
echo "📋 Next Steps:"
echo "1. Deploy your application code to /opt/yitro-crm"
echo "2. Run the deployment script"
echo "3. Configure SSL with Certbot"
echo ""
echo "💡 Database Details:"
echo "   - Database: yitro_crm_prod"
echo "   - User: yitro_db_user"
echo "   - Password: YitroSecure123!"
echo ""
echo "🔐 Save this JWT secret securely:"
echo "$JWT_SECRET"
