import express, { type Express } from "express";
import cors from "cors";
import { handleDemo } from "./routes/demo";
import * as crmRoutes from "./routes/crm-prisma";
import authRoutes from "./routes/auth";
import adminRoutes from "./routes/admin";
import { initializeDatabase } from "./db/init-db";
import { setupTestAccounts } from "./db/setup-test-accounts";

export function createServer(): Express {
  const app = express();

  // Middleware
  app.use(cors());
  app.use(express.json());
  app.use(express.urlencoded({ extended: true }));

  // Debug middleware
  app.use((req, res, next) => {
    if (req.url.startsWith("/api/")) {
      console.log(`API Request: ${req.method} ${req.url}`);
      console.log(`Request params:`, req.params);
      console.log(`Request body:`, req.body);
    }
    next();
  });

  // Initialize sample data (only if database is available)
  crmRoutes.initializeSampleData().catch((error) => {
    console.log("📝 Setting up in-memory test accounts...");
    console.log("✅ In-memory test accounts ready!");
    console.log("🔑 Admin Login - Email: admin@yitro.com | Password: admin123");
    console.log("🔑 User Login - Email: user@yitro.com | Password: user123");
  });

  // Initialize authentication database
  initializeDatabase()
    .then(() => {
      // Setup test accounts after database is ready
      setupTestAccounts().catch(console.error);
    })
    .catch(console.error);

  // Auth API Routes
  app.use("/api/auth", authRoutes);
  app.use("/api/admin", adminRoutes);

  // CRM API Routes

  // Contacts
  app.get("/api/contacts", crmRoutes.getContacts);
  app.get("/api/contacts/:id", crmRoutes.getContact);
  app.post("/api/contacts", crmRoutes.createContact);
  app.put("/api/contacts/:id", crmRoutes.updateContact);
  app.delete("/api/contacts/:id", crmRoutes.deleteContact);

  // Accounts
  app.get("/api/accounts", crmRoutes.getAccounts);
  app.get("/api/accounts/:id", crmRoutes.getAccount);
  app.post("/api/accounts", crmRoutes.createAccount);
  app.put("/api/accounts/:id", crmRoutes.updateAccount);
  app.delete("/api/accounts/:id", crmRoutes.deleteAccount);

  // Activities
  app.get("/api/activities", crmRoutes.getActivities);
  app.get("/api/activities/:id", crmRoutes.getActivity);
  app.post("/api/activities", crmRoutes.createActivity);
  app.put("/api/activities/:id", crmRoutes.updateActivity);
  app.delete("/api/activities/:id", crmRoutes.deleteActivity);

  // Deals
  app.get("/api/deals", crmRoutes.getDeals);
  app.get("/api/deals/:id", crmRoutes.getDeal);
  app.post("/api/deals", crmRoutes.createDeal);
  app.put("/api/deals/:id", crmRoutes.updateDeal);
  app.delete("/api/deals/:id", crmRoutes.deleteDeal);

  // Leads
  app.get("/api/leads", crmRoutes.getLeads);
  app.get("/api/leads/:id", crmRoutes.getLead);
  app.post("/api/leads", crmRoutes.createLead);
  app.put("/api/leads/:id", crmRoutes.updateLead);
  app.delete("/api/leads/:id", crmRoutes.deleteLead);

  // Reports
  app.post("/api/reports/generate", crmRoutes.generateReport);
  app.get("/api/reports/download/:id", crmRoutes.downloadReport);

  // User Profile
  app.get("/api/profile", crmRoutes.getCurrentUserProfile);
  app.put("/api/profile", crmRoutes.updateCurrentUserProfile);
  app.get("/api/profile/:id", crmRoutes.getUserProfile);
  app.put("/api/profile/:id", crmRoutes.updateUserProfile);

  // Existing routes
  app.get("/api/ping", (req, res) => {
    res.json({ message: "pong" });
  });

  app.get("/api/demo", handleDemo);

  return app;
}
