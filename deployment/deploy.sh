#!/bin/bash

# Yitro CRM Deployment Script
# Usage: ./deploy.sh [branch] [--setup]

set -e

# Configuration
REPO_URL="https://github.com/yourusername/yitro-crm.git"  # Update with your repo URL
APP_DIR="/opt/yitro-crm"
APP_USER="yitro"
NGINX_SITE="dealhub.yitrobc.net"
BRANCH="${1:-main}"
SETUP_MODE="$2"

echo "🚀 Deploying Yitro CRM to VPS..."
echo "================================"
echo "📦 Branch: $BRANCH"
echo "📁 Directory: $APP_DIR"
echo "👤 User: $APP_USER"
echo ""

# Check if this is initial setup
if [[ "$SETUP_MODE" == "--setup" ]]; then
    echo "🔧 Running initial setup..."
    
    # Clone repository
    echo "📥 Cloning repository..."
    if [ -d "$APP_DIR" ]; then
        rm -rf "$APP_DIR"
    fi
    git clone "$REPO_URL" "$APP_DIR"
    cd "$APP_DIR"
    git checkout "$BRANCH"
    
    # Set ownership
    chown -R $APP_USER:$APP_USER "$APP_DIR"
    
    # Copy deployment files
    echo "📋 Setting up configuration files..."
    
    # Copy Nginx configuration
    cp deployment/nginx-dealhub.conf /etc/nginx/sites-available/$NGINX_SITE
    ln -sf /etc/nginx/sites-available/$NGINX_SITE /etc/nginx/sites-enabled/
    rm -f /etc/nginx/sites-enabled/default
    
    # Test Nginx configuration
    nginx -t
    
    # Copy PM2 ecosystem
    cp deployment/ecosystem.config.js "$APP_DIR/"
    chown $APP_USER:$APP_USER "$APP_DIR/ecosystem.config.js"
    
else
    echo "🔄 Updating existing deployment..."
    cd "$APP_DIR"
    
    # Stop application
    echo "⏹️  Stopping application..."
    sudo -u $APP_USER pm2 stop yitro-crm || echo "App not running"
    
    # Pull latest changes
    echo "📥 Pulling latest changes..."
    git fetch origin
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
    
    # Set ownership
    chown -R $APP_USER:$APP_USER "$APP_DIR"
fi

# Switch to app user for remaining operations
echo "👤 Switching to application user..."

# Install dependencies and build
sudo -u $APP_USER bash << 'EOF'
cd /opt/yitro-crm

# Install dependencies
echo "📦 Installing dependencies..."
npm ci --production=false

# Generate Prisma client
echo "🗄️  Generating Prisma client..."
npx prisma generate

# Run database migrations
echo "🔄 Running database migrations..."
npx prisma migrate deploy

# Build application
echo "🔨 Building application..."
npm run build

# Install PM2 if not already installed globally
if ! command -v pm2 &> /dev/null; then
    echo "⚡ Installing PM2..."
    npm install -g pm2
fi
EOF

# Setup SSL certificate (only on initial setup)
if [[ "$SETUP_MODE" == "--setup" ]]; then
    echo "🔒 Setting up SSL certificate..."
    certbot --nginx -d dealhub.yitrobc.net -d www.dealhub.yitrobc.net --non-interactive --agree-tos -m admin@yitrobc.net
fi

# Start/restart application
echo "🚀 Starting application..."
sudo -u $APP_USER pm2 reload ecosystem.config.js
sudo -u $APP_USER pm2 save

# Setup PM2 startup (only on initial setup)
if [[ "$SETUP_MODE" == "--setup" ]]; then
    echo "⚡ Setting up PM2 startup..."
    env PATH=$PATH:/usr/bin pm2 startup systemd -u $APP_USER --hp /home/$APP_USER
fi

# Reload Nginx
echo "🌐 Reloading Nginx..."
systemctl reload nginx

# Show status
echo ""
echo "📊 Application Status:"
sudo -u $APP_USER pm2 status

echo ""
echo "🌐 Nginx Status:"
systemctl status nginx --no-pager -l

echo ""
echo "🎉 Deployment Complete!"
echo "======================"
echo ""
echo "🌍 Your app is now running at:"
echo "   https://dealhub.yitrobc.net"
echo ""
echo "📊 Management Commands:"
echo "   sudo -u $APP_USER pm2 status     # Check app status"
echo "   sudo -u $APP_USER pm2 logs       # View logs"
echo "   sudo -u $APP_USER pm2 restart yitro-crm  # Restart app"
echo ""
echo "🔧 Logs Location:"
echo "   Application: /var/log/yitro-crm/"
echo "   Nginx: /var/log/nginx/"
echo ""
echo "✅ Login credentials:"
echo "   Admin: admin@yitro.com / admin123"
echo "   User: user@yitro.com / user123"
