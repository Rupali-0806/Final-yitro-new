import serverless from "serverless-http";
import { createServer } from "../../server";

// Initialize the Express server
const app = createServer();

// Export the serverless handler with production configuration
export const handler = serverless(app, {
  binary: false,
  request: (request: any, event: any, context: any) => {
    // Ensure production environment
    process.env.NODE_ENV = process.env.NODE_ENV || "production";
  },
});
