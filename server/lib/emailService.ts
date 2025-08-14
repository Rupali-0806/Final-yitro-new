import nodemailer from "nodemailer";

export interface EmailNotification {
  to: string;
  subject: string;
  html: string;
  text?: string;
}

class EmailService {
  private transporter: nodemailer.Transporter | null = null;

  constructor() {
    // Don't initialize transporter in constructor to avoid issues during module loading
  }

  private getTransporter(): nodemailer.Transporter {
    if (!this.transporter) {
      this.transporter = this.createTransporter();
    }
    return this.transporter;
  }

  private createTransporter(): nodemailer.Transporter {
    const smtpService = process.env.SMTP_SERVICE?.toLowerCase();
    const smtpHost = process.env.SMTP_HOST;
    const smtpPort = parseInt(process.env.SMTP_PORT || "587");
    const smtpSecure = process.env.SMTP_SECURE === "true";
    const smtpUser = process.env.SMTP_USER;
    const smtpPassword = process.env.SMTP_PASSWORD;

    // If no SMTP configuration is provided, use a test transporter
    if (!smtpUser || !smtpPassword) {
      console.log(
        "⚠️ No SMTP configuration found. Using test transporter (emails will not be sent).",
      );
      return nodemailer.createTransporter({
        host: "smtp.ethereal.email",
        port: 587,
        secure: false,
        auth: {
          user: "test@example.com",
          pass: "test-password",
        },
      });
    }

    // Configure based on service or custom SMTP
    if (smtpService) {
      // Use predefined service configurations
      switch (smtpService) {
        case "gmail":
          return nodemailer.createTransporter({
            service: "gmail",
            auth: {
              user: smtpUser,
              pass: smtpPassword,
            },
          });

        case "sendgrid":
          return nodemailer.createTransporter({
            host: "smtp.sendgrid.net",
            port: 587,
            secure: false,
            auth: {
              user: "apikey",
              pass: smtpPassword, // This should be your SendGrid API key
            },
          });

        case "mailgun":
          return nodemailer.createTransporter({
            host: "smtp.mailgun.org",
            port: 587,
            secure: false,
            auth: {
              user: smtpUser,
              pass: smtpPassword,
            },
          });

        case "ses":
        case "aws":
          return nodemailer.createTransporter({
            host:
              process.env.AWS_SES_HOST || "email-smtp.us-east-1.amazonaws.com",
            port: 587,
            secure: false,
            auth: {
              user: smtpUser, // AWS SES SMTP username
              pass: smtpPassword, // AWS SES SMTP password
            },
          });

        case "postmark":
          return nodemailer.createTransporter({
            host: "smtp.postmarkapp.com",
            port: 587,
            secure: false,
            auth: {
              user: smtpUser,
              pass: smtpPassword, // Postmark Server API Token
            },
          });

        case "outlook":
        case "hotmail":
          return nodemailer.createTransporter({
            service: "hotmail",
            auth: {
              user: smtpUser,
              pass: smtpPassword,
            },
          });

        default:
          console.log(
            `Unknown SMTP service: ${smtpService}. Using custom SMTP configuration.`,
          );
        // Fall through to custom configuration
      }
    }

    // Use custom SMTP configuration
    if (smtpHost) {
      return nodemailer.createTransporter({
        host: smtpHost,
        port: smtpPort,
        secure: smtpSecure, // true for 465, false for other ports
        auth: {
          user: smtpUser,
          pass: smtpPassword,
        },
        tls: {
          // Do not fail on invalid certs for development
          rejectUnauthorized: process.env.NODE_ENV === "production",
        },
      });
    }

    // Fallback to Gmail service if no specific configuration
    console.log("Using Gmail as fallback SMTP service.");
    return nodemailer.createTransporter({
      service: "gmail",
      auth: {
        user: smtpUser,
        pass: smtpPassword,
      },
    });
  }

  // Send welcome email
  public async sendWelcomeEmail(
    email: string,
    displayName: string,
    verificationToken: string,
  ): Promise<void> {
    const verificationUrl = `${process.env.FRONTEND_URL || "http://localhost:8080"}/verify-email?token=${verificationToken}`;

    const html = `
      <!DOCTYPE html>
      <html>
      <head>
        <style>
          body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
          .container { max-width: 600px; margin: 0 auto; padding: 20px; }
          .header { background: linear-gradient(135deg, #60A5FA, #3B82F6); color: white; padding: 30px; text-align: center; border-radius: 8px 8px 0 0; }
          .content { background: #f8fafc; padding: 30px; border-radius: 0 0 8px 8px; }
          .button { display: inline-block; background: #3B82F6; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; margin: 20px 0; }
          .button:hover { background: #2563EB; }
          .footer { text-align: center; margin-top: 30px; color: #6B7280; font-size: 14px; }
        </style>
      </head>
      <body>
        <div class="container">
          <div class="header">
            <h1>Welcome to Yitro CRM! 🎉</h1>
            <p>Your account has been created successfully</p>
          </div>
          <div class="content">
            <h2>Hello ${displayName}!</h2>
            <p>Welcome to Yitro CRM platform. To get started, please verify your email address by clicking the button below:</p>
            
            <div style="text-align: center;">
              <a href="${verificationUrl}" class="button">Verify Your Email</a>
            </div>
            
            <p><strong>Security Note:</strong> This verification link will expire in 24 hours for your security.</p>
            
            <h3>What's Next?</h3>
            <ul>
              <li>✅ Verify your email address</li>
              <li>🏠 Access your personalized dashboard</li>
              <li>👥 Start managing contacts and deals</li>
              <li>📊 Track your sales performance</li>
            </ul>
            
            <p>If you didn't create this account, please ignore this email.</p>
          </div>
          <div class="footer">
            <p>© 2024 Yitro CRM. All rights reserved.</p>
            <p>If you have any questions, reply to this email or contact our support team.</p>
          </div>
        </div>
      </body>
      </html>
    `;

    await this.sendEmail({
      to: email,
      subject: "🎉 Welcome to Yitro CRM - Verify Your Email",
      html,
      text: `Welcome to Yitro CRM! Please verify your email by visiting: ${verificationUrl}`,
    });
  }

  // Send login notification
  public async sendLoginNotification(
    email: string,
    displayName: string,
    loginDetails: { ip?: string; userAgent?: string; timestamp: Date },
  ): Promise<void> {
    const html = `
      <!DOCTYPE html>
      <html>
      <head>
        <style>
          body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
          .container { max-width: 600px; margin: 0 auto; padding: 20px; }
          .header { background: #10B981; color: white; padding: 20px; text-align: center; border-radius: 8px 8px 0 0; }
          .content { background: #f8fafc; padding: 30px; border-radius: 0 0 8px 8px; }
          .alert { background: #FEF3C7; border: 1px solid #F59E0B; padding: 15px; border-radius: 6px; margin: 20px 0; }
          .details { background: white; padding: 15px; border-radius: 6px; margin: 15px 0; }
          .footer { text-align: center; margin-top: 30px; color: #6B7280; font-size: 14px; }
        </style>
      </head>
      <body>
        <div class="container">
          <div class="header">
            <h1>🔐 Login Detected</h1>
            <p>Someone signed into your Yitro CRM account</p>
          </div>
          <div class="content">
            <h2>Hello ${displayName},</h2>
            <p>We detected a new sign-in to your Yitro CRM account. Here are the details:</p>
            
            <div class="details">
              <h3>Login Details:</h3>
              <ul>
                <li><strong>Time:</strong> ${loginDetails.timestamp.toLocaleString()}</li>
                <li><strong>IP Address:</strong> ${loginDetails.ip || "Unknown"}</li>
                <li><strong>Device:</strong> ${loginDetails.userAgent || "Unknown"}</li>
              </ul>
            </div>
            
            <div class="alert">
              <h3>⚠️ Security Alert</h3>
              <p><strong>Was this you?</strong></p>
              <p>If you recognize this login, no action is needed.</p>
              <p>If you didn't sign in, please immediately:</p>
              <ul>
                <li>Change your password</li>
                <li>Contact our support team</li>
                <li>Review your account activity</li>
              </ul>
            </div>
            
            <p>For your security, we recommend:</p>
            <ul>
              <li>🔒 Using a strong, unique password</li>
              <li>🔐 Enabling two-factor authentication</li>
              <li>💻 Signing out of unused devices</li>
            </ul>
          </div>
          <div class="footer">
            <p>© 2024 Yitro CRM. All rights reserved.</p>
            <p>This is an automated security notification.</p>
          </div>
        </div>
      </body>
      </html>
    `;

    await this.sendEmail({
      to: email,
      subject: "🔐 Login Alert - Yitro CRM Account Access",
      html,
      text: `Login detected for your Yitro CRM account at ${loginDetails.timestamp.toLocaleString()}. If this wasn't you, please secure your account immediately.`,
    });
  }

  // Send password reset email
  public async sendPasswordResetEmail(
    email: string,
    displayName: string,
    resetToken: string,
  ): Promise<void> {
    const resetUrl = `${process.env.FRONTEND_URL || "http://localhost:8080"}/reset-password?token=${resetToken}`;

    const html = `
      <!DOCTYPE html>
      <html>
      <head>
        <style>
          body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
          .container { max-width: 600px; margin: 0 auto; padding: 20px; }
          .header { background: #DC2626; color: white; padding: 20px; text-align: center; border-radius: 8px 8px 0 0; }
          .content { background: #f8fafc; padding: 30px; border-radius: 0 0 8px 8px; }
          .button { display: inline-block; background: #DC2626; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; margin: 20px 0; }
          .button:hover { background: #B91C1C; }
          .alert { background: #FEE2E2; border: 1px solid #DC2626; padding: 15px; border-radius: 6px; margin: 20px 0; }
          .footer { text-align: center; margin-top: 30px; color: #6B7280; font-size: 14px; }
        </style>
      </head>
      <body>
        <div class="container">
          <div class="header">
            <h1>🔑 Password Reset Request</h1>
            <p>Reset your Yitro CRM password</p>
          </div>
          <div class="content">
            <h2>Hello ${displayName},</h2>
            <p>We received a request to reset your password for your Yitro CRM account.</p>
            
            <div style="text-align: center;">
              <a href="${resetUrl}" class="button">Reset Your Password</a>
            </div>
            
            <div class="alert">
              <h3>⚠️ Important Security Information</h3>
              <ul>
                <li>This link will expire in <strong>1 hour</strong></li>
                <li>If you didn't request this reset, ignore this email</li>
                <li>Your current password remains unchanged until you complete the reset</li>
              </ul>
            </div>
            
            <p>For security reasons, this password reset link can only be used once.</p>
            
            <p>If you continue to have problems, please contact our support team.</p>
          </div>
          <div class="footer">
            <p>© 2024 Yitro CRM. All rights reserved.</p>
            <p>This is an automated security notification.</p>
          </div>
        </div>
      </body>
      </html>
    `;

    await this.sendEmail({
      to: email,
      subject: "🔑 Password Reset - Yitro CRM",
      html,
      text: `Reset your Yitro CRM password by visiting: ${resetUrl}. This link expires in 1 hour.`,
    });
  }

  // Send employee welcome email (admin-created accounts)
  public async sendEmployeeWelcomeEmail(
    email: string,
    displayName: string,
    password: string,
    verificationToken: string,
  ): Promise<void> {
    const verificationUrl = `${process.env.FRONTEND_URL || "http://localhost:8080"}/verify-email?token=${verificationToken}`;

    const html = `
      <!DOCTYPE html>
      <html>
      <head>
        <style>
          body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
          .container { max-width: 600px; margin: 0 auto; padding: 20px; }
          .header { background: linear-gradient(135deg, #60A5FA, #3B82F6); color: white; padding: 30px; text-align: center; border-radius: 8px 8px 0 0; }
          .content { background: #f8fafc; padding: 30px; border-radius: 0 0 8px 8px; }
          .credentials { background: white; border: 2px solid #3B82F6; padding: 20px; border-radius: 6px; margin: 20px 0; }
          .button { display: inline-block; background: #3B82F6; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; margin: 20px 0; }
          .button:hover { background: #2563EB; }
          .footer { text-align: center; margin-top: 30px; color: #6B7280; font-size: 14px; }
          .warning { background: #FEF3C7; border: 1px solid #F59E0B; padding: 15px; border-radius: 6px; margin: 20px 0; }
        </style>
      </head>
      <body>
        <div class="container">
          <div class="header">
            <h1>Welcome to Yitro CRM! 🎉</h1>
            <p>Your company account has been created</p>
          </div>
          <div class="content">
            <h2>Hello ${displayName}!</h2>
            <p>Your administrator has created a Yitro CRM account for you. Below are your login credentials:</p>

            <div class="credentials">
              <h3>🔐 Your Login Credentials</h3>
              <p><strong>Email:</strong> ${email}</p>
              <p><strong>Password:</strong> <code style="background: #f3f4f6; padding: 4px 8px; border-radius: 4px; font-family: monospace;">${password}</code></p>
            </div>

            <div class="warning">
              <h3>⚠️ Important Security Steps</h3>
              <ol>
                <li>Click the verification button below to activate your account</li>
                <li>Login with the provided credentials</li>
                <li>Change your password immediately for security</li>
                <li>Keep your login credentials secure</li>
              </ol>
            </div>

            <div style="text-align: center;">
              <a href="${verificationUrl}" class="button">Verify Your Account</a>
            </div>

            <h3>What's Next?</h3>
            <ul>
              <li>✅ Verify your account (required)</li>
              <li>🔑 Login with provided credentials</li>
              <li>🔒 Change your password for security</li>
              <li>🏠 Access your personalized CRM dashboard</li>
              <li>👥 Start managing your contacts and deals</li>
            </ul>

            <p><strong>Need Help?</strong> Contact your administrator or IT support team.</p>
          </div>
          <div class="footer">
            <p>© 2024 Yitro CRM. All rights reserved.</p>
            <p>This account was created by your company administrator.</p>
          </div>
        </div>
      </body>
      </html>
    `;

    await this.sendEmail({
      to: email,
      subject: "🎉 Welcome to Yitro CRM - Your Account Details",
      html,
      text: `Welcome to Yitro CRM! Your login credentials: Email: ${email}, Password: ${password}. Please verify your account at: ${verificationUrl}`,
    });
  }

  // Test email configuration
  public async testConnection(): Promise<boolean> {
    try {
      console.log("🧪 Testing SMTP connection...");
      const transporter = this.getTransporter();
      await transporter.verify();
      console.log("✅ SMTP connection successful!");
      return true;
    } catch (error) {
      console.error("❌ SMTP connection failed:", error);
      return false;
    }
  }

  // Send test email
  public async sendTestEmail(toEmail: string): Promise<void> {
    const html = `
      <!DOCTYPE html>
      <html>
      <head>
        <style>
          body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
          .container { max-width: 600px; margin: 0 auto; padding: 20px; }
          .header { background: #10B981; color: white; padding: 20px; text-align: center; border-radius: 8px 8px 0 0; }
          .content { background: #f8fafc; padding: 30px; border-radius: 0 0 8px 8px; }
          .footer { text-align: center; margin-top: 30px; color: #6B7280; font-size: 14px; }
        </style>
      </head>
      <body>
        <div class="container">
          <div class="header">
            <h1>🧪 SMTP Test Email</h1>
            <p>Your email configuration is working!</p>
          </div>
          <div class="content">
            <h2>Congratulations!</h2>
            <p>This is a test email to verify that your SMTP configuration is working correctly.</p>

            <h3>Configuration Details:</h3>
            <ul>
              <li><strong>SMTP Service:</strong> ${process.env.SMTP_SERVICE || "Gmail (default)"}</li>
              <li><strong>SMTP User:</strong> ${process.env.SMTP_USER || "Not configured"}</li>
              <li><strong>Timestamp:</strong> ${new Date().toLocaleString()}</li>
            </ul>

            <p>Your Yitro CRM application is now ready to send emails for:</p>
            <ul>
              <li>✅ User registration and verification</li>
              <li>✅ Password reset notifications</li>
              <li>✅ Login security alerts</li>
              <li>✅ Employee welcome emails</li>
            </ul>
          </div>
          <div class="footer">
            <p>© 2024 Yitro CRM. All rights reserved.</p>
            <p>This is an automated test email from your CRM system.</p>
          </div>
        </div>
      </body>
      </html>
    `;

    await this.sendEmail({
      to: toEmail,
      subject: "🧪 Yitro CRM - SMTP Configuration Test",
      html,
      text: `SMTP Configuration Test - Your email configuration is working! Timestamp: ${new Date().toLocaleString()}`,
    });
  }

  // Send generic email
  private async sendEmail(emailData: EmailNotification): Promise<void> {
    try {
      const transporter = this.getTransporter();
      const info = await transporter.sendMail({
        from: `"Yitro CRM" <${process.env.SMTP_USER || "noreply@yitro.com"}>`,
        to: emailData.to,
        subject: emailData.subject,
        text: emailData.text,
        html: emailData.html,
      });

      console.log("✅ Email sent successfully:", info.messageId);
    } catch (error) {
      console.error("❌ Email sending error:", error);
      throw new Error("Failed to send email");
    }
  }
}

export const emailService = new EmailService();
