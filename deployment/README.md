# VPS Deployment Guide for dealhub.yitrobc.net

## Server Details

- **Server**: 216.48.184.73
- **Domain**: https://dealhub.yitrobc.net
- **SSH**: `ssh root@216.48.184.73`

## Quick Deployment Steps

### 1. Connect to Your Server

```bash
ssh root@216.48.184.73
# Password: ABZUZG@ywgdb581
```

### 2. Upload Deployment Files

Upload the `deployment/` folder to your server:

```bash
# On your local machine
scp -r deployment/ root@216.48.184.73:/root/
```

### 3. Run Initial Server Setup

```bash
# On the server
cd /root/deployment
chmod +x setup-vps.sh
./setup-vps.sh
```

### 4. Update Repository URL

Edit the deployment script with your actual GitHub repository:

```bash
nano deploy.sh
# Update REPO_URL to your actual repository URL
```

### 5. Run Initial Deployment

```bash
chmod +x deploy.sh
./deploy.sh main --setup
```

## What Gets Installed

### Software Stack

- ✅ Node.js 20.x
- ✅ PostgreSQL 15
- ✅ Nginx web server
- ✅ PM2 process manager
- ✅ Certbot for SSL certificates

### Database Setup

- **Database**: `yitro_crm_prod`
- **User**: `yitro_db_user`
- **Password**: `YitroSecure123!`

### Application Structure

```
/opt/yitro-crm/           # Application directory
├── dist/                 # Built application
├── .env                  # Environment variables
└── ecosystem.config.js   # PM2 configuration

/var/log/yitro-crm/       # Application logs
├── error.log
├── out.log
└── combined.log
```

## Environment Variables

The setup script automatically creates `/opt/yitro-crm/.env` with:

```env
NODE_ENV=production
PORT=3000
DATABASE_URL=postgresql://yitro_db_user:YitroSecure123!@localhost:5432/yitro_crm_prod
STACK_SECRET_SERVER_KEY=[auto-generated]
FRONTEND_URL=https://dealhub.yitrobc.net

# Update these for email functionality
SMTP_SERVICE=gmail
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
```

## Management Commands

### Application Management

```bash
# Check status
sudo -u yitro pm2 status

# View logs
sudo -u yitro pm2 logs

# Restart application
sudo -u yitro pm2 restart yitro-crm

# Stop application
sudo -u yitro pm2 stop yitro-crm
```

### Update Deployment

```bash
cd /root/deployment
./deploy.sh main
```

### Nginx Management

```bash
# Check status
systemctl status nginx

# Reload configuration
systemctl reload nginx

# View logs
tail -f /var/log/nginx/dealhub.yitrobc.net.access.log
```

### Database Management

```bash
# Connect to database
sudo -u postgres psql -d yitro_crm_prod

# View database logs
tail -f /var/log/postgresql/postgresql-15-main.log
```

## SSL Certificate

SSL certificate is automatically set up during initial deployment using Let's Encrypt.

**Certificate renewal** (automatic):

```bash
# Test renewal
certbot renew --dry-run

# Check certificate status
certbot certificates
```

## Security Features

- ✅ Firewall configured (UFW)
- ✅ SSL/TLS encryption
- ✅ Security headers
- ✅ Gzip compression
- ✅ Rate limiting
- ✅ Environment variable protection

## Troubleshooting

### Check Application Logs

```bash
sudo -u yitro pm2 logs yitro-crm
tail -f /var/log/yitro-crm/error.log
```

### Check Nginx Logs

```bash
tail -f /var/log/nginx/dealhub.yitrobc.net.error.log
```

### Check Database Connection

```bash
sudo -u postgres psql -d yitro_crm_prod -c "SELECT version();"
```

### Restart All Services

```bash
systemctl restart nginx
sudo -u yitro pm2 restart yitro-crm
```

## Login Credentials

After deployment, you can login with:

- **Admin**: admin@yitro.com / admin123
- **User**: user@yitro.com / user123

**Important**: Change the default admin password after first login!

## Support

For deployment issues:

1. Check application logs: `/var/log/yitro-crm/`
2. Check Nginx logs: `/var/log/nginx/`
3. Verify database connection
4. Check PM2 process status

---

🎉 **Your Yitro CRM will be live at https://dealhub.yitrobc.net**
