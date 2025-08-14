import { Router, RequestHandler } from "express";
import { llmService } from "../lib/llmService";
import { requireAuth } from "../lib/neonAuth";

const router = Router();

interface ChatRequest {
  query: string;
  crmData: {
    leads: any[];
    accounts: any[];
    contacts: any[];
    deals: any[];
  };
  conversationHistory?: Array<{ role: "user" | "assistant"; content: string }>;
  user?: {
    displayName?: string;
    email?: string;
    role?: string;
  };
}

interface ChatResponse {
  success: boolean;
  data?: {
    message: string;
    intent?: string;
    quickActions?: string[];
  };
  error?: string;
}

// Chat endpoint with LLM integration
const handleChatQuery: RequestHandler<{}, ChatResponse, ChatRequest> = async (
  req,
  res,
) => {
  try {
    const { query, crmData, conversationHistory = [], user } = req.body;

    if (!query) {
      return res.status(400).json({
        success: false,
        error: "Query is required",
      });
    }

    if (!crmData) {
      return res.status(400).json({
        success: false,
        error: "CRM data is required for context",
      });
    }

    console.log(`🤖 Processing LLM chat query: "${query.substring(0, 50)}..."`);

    // Prepare CRM context for LLM
    const crmContext = {
      leads: crmData.leads || [],
      accounts: crmData.accounts || [],
      contacts: crmData.contacts || [],
      deals: crmData.deals || [],
      user: user || {},
    };

    // Generate response using LLM service
    const response = await llmService.generateResponse(
      query,
      crmContext,
      conversationHistory,
    );

    console.log(`✅ LLM response generated successfully`);

    res.json({
      success: true,
      data: {
        message: response.message,
        intent: response.intent,
        quickActions: response.quickActions,
      },
    });
  } catch (error: any) {
    console.error("LLM Chat Error:", error);
    res.status(500).json({
      success: false,
      error: error.message || "Failed to process chat query",
    });
  }
};

// Test LLM connection endpoint
const testLLMConnection: RequestHandler = async (req, res) => {
  try {
    const result = await llmService.testConnection();

    res.json({
      success: result.success,
      message: result.message,
    });
  } catch (error: any) {
    res.status(500).json({
      success: false,
      error: error.message || "Failed to test LLM connection",
    });
  }
};

// Routes
router.post("/chat", handleChatQuery);
router.get("/test-connection", testLLMConnection);

export default router;
