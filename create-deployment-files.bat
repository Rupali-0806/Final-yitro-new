@echo off
echo Creating deployment files for Windows...

mkdir deployment 2>nul

echo Creating setup-vps.sh...
(
echo #!/bin/bash
echo.
echo # Yitro CRM VPS Setup Script
echo # Run this on your Ubuntu/Debian VPS server
echo.
echo set -e
echo.
echo echo "🚀 Setting up Yitro CRM on VPS..."
echo echo "=================================="
echo.
echo # Update system
echo echo "📦 Updating system packages..."
echo apt update ^&^& apt upgrade -y
echo.
echo # Install essential packages
echo echo "🔧 Installing essential packages..."
echo apt install -y curl wget git build-essential software-properties-common
echo.
echo # Install Node.js 20.x
echo echo "📱 Installing Node.js 20.x..."
echo curl -fsSL https://deb.nodesource.com/setup_20.x ^| bash -
echo apt install -y nodejs
echo.
echo # Install PostgreSQL
echo echo "🗄️  Installing PostgreSQL..."
echo apt install -y postgresql postgresql-contrib
echo systemctl start postgresql
echo systemctl enable postgresql
echo.
echo # Install Nginx
echo echo "🌐 Installing Nginx..."
echo apt install -y nginx
echo systemctl start nginx
echo systemctl enable nginx
echo.
echo # Install PM2
echo echo "⚡ Installing PM2..."
echo npm install -g pm2
echo.
echo # Install Certbot
echo echo "🔒 Installing Certbot..."
echo apt install -y certbot python3-certbot-nginx
echo.
echo # Create app user
echo echo "👤 Creating application user..."
echo useradd -m -s /bin/bash yitro ^|^| echo "User already exists"
echo.
echo # Create directories
echo mkdir -p /opt/yitro-crm
echo chown yitro:yitro /opt/yitro-crm
echo mkdir -p /var/log/yitro-crm
echo chown yitro:yitro /var/log/yitro-crm
echo.
echo # Setup database
echo echo "🗄️  Setting up PostgreSQL..."
echo sudo -u postgres psql ^<^< 'EOF'
echo CREATE USER yitro_db_user WITH PASSWORD 'YitroSecure123!'^;
echo CREATE DATABASE yitro_crm_prod OWNER yitro_db_user^;
echo GRANT ALL PRIVILEGES ON DATABASE yitro_crm_prod TO yitro_db_user^;
echo \q
echo EOF
echo.
echo # Generate JWT secret and create .env
echo JWT_SECRET=$^(openssl rand -hex 64^)
echo cat ^> /opt/yitro-crm/.env ^<^< EOF
echo NODE_ENV=production
echo PORT=3000
echo DATABASE_URL=postgresql://yitro_db_user:YitroSecure123!@localhost:5432/yitro_crm_prod
echo STACK_SECRET_SERVER_KEY=$JWT_SECRET
echo FRONTEND_URL=https://dealhub.yitrobc.net
echo SMTP_SERVICE=gmail
echo SMTP_USER=your-email@gmail.com
echo SMTP_PASSWORD=your-app-password
echo EOF
echo chown yitro:yitro /opt/yitro-crm/.env
echo chmod 600 /opt/yitro-crm/.env
echo.
echo # Configure firewall
echo ufw allow ssh
echo ufw allow 'Nginx Full'
echo ufw --force enable
echo.
echo echo "🎉 VPS Setup Complete!"
) > deployment\setup-vps.sh

echo Creating deploy.sh...
(
echo #!/bin/bash
echo set -e
echo REPO_URL="https://github.com/Rupali-0806/Final-yitro-new.git"
echo APP_DIR="/opt/yitro-crm"
echo APP_USER="yitro"
echo NGINX_SITE="dealhub.yitrobc.net"
echo BRANCH="${1:-spark-oasis}"
echo SETUP_MODE="$2"
echo.
echo echo "🚀 Deploying Yitro CRM..."
echo.
echo if [[ "$SETUP_MODE" == "--setup" ]]; then
echo   echo "🔧 Initial setup..."
echo   if [ -d "$APP_DIR" ]; then rm -rf "$APP_DIR"; fi
echo   git clone "$REPO_URL" "$APP_DIR"
echo   cd "$APP_DIR"
echo   git checkout "$BRANCH"
echo   chown -R $APP_USER:$APP_USER "$APP_DIR"
echo   cp deployment/nginx-dealhub.conf /etc/nginx/sites-available/$NGINX_SITE
echo   ln -sf /etc/nginx/sites-available/$NGINX_SITE /etc/nginx/sites-enabled/
echo   rm -f /etc/nginx/sites-enabled/default
echo   nginx -t
echo   cp deployment/ecosystem.config.js "$APP_DIR/"
echo   chown $APP_USER:$APP_USER "$APP_DIR/ecosystem.config.js"
echo else
echo   echo "🔄 Updating..."
echo   cd "$APP_DIR"
echo   sudo -u $APP_USER pm2 stop yitro-crm ^|^| echo "Not running"
echo   git fetch origin
echo   git checkout "$BRANCH"
echo   git pull origin "$BRANCH"
echo   chown -R $APP_USER:$APP_USER "$APP_DIR"
echo fi
echo.
echo # Build and deploy
echo sudo -u $APP_USER bash ^<^< 'EOF'
echo cd /opt/yitro-crm
echo npm ci --production=false
echo npx prisma generate
echo npx prisma migrate deploy
echo npm run build
echo EOF
echo.
echo if [[ "$SETUP_MODE" == "--setup" ]]; then
echo   certbot --nginx -d dealhub.yitrobc.net --non-interactive --agree-tos -m admin@yitrobc.net
echo fi
echo.
echo sudo -u $APP_USER pm2 reload ecosystem.config.js
echo sudo -u $APP_USER pm2 save
echo systemctl reload nginx
echo.
echo echo "🎉 Deployment Complete! Visit https://dealhub.yitrobc.net"
) > deployment\deploy.sh

echo Deployment files created successfully!
echo.
echo Next steps:
echo 1. Upload to VPS: scp -r deployment/ root@216.48.184.73:/root/
echo 2. SSH to server: ssh root@216.48.184.73
echo 3. Run: cd /root/deployment && chmod +x *.sh && ./setup-vps.sh
echo 4. Then: ./deploy.sh spark-oasis --setup
