import express, { type Express } from "express";
import cors from "cors";
import quantumRoutes from "./quantum/routes";

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
    }
    next();
  });

  // Quantum Healthcare API Routes
  app.use("/api/quantum", quantumRoutes);

  // Health check
  app.get("/api/ping", (req, res) => {
    res.json({ message: "pong", service: "quantum-healthcare" });
  });

  app.get("/api/demo", (req, res) => {
    res.json({ message: "Quantum Healthcare AI is ready!" });
  });

  return app;
}