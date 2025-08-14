interface LLMChatRequest {
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

interface LLMChatResponse {
  success: boolean;
  data?: {
    message: string;
    intent?: string;
    quickActions?: string[];
  };
  error?: string;
}

export class LLMChatAPI {
  private baseURL: string;

  constructor() {
    this.baseURL = import.meta.env.VITE_API_URL || "/api";
  }

  async sendChatQuery(request: LLMChatRequest): Promise<LLMChatResponse> {
    try {
      const response = await fetch(`${this.baseURL}/llm/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        credentials: "include",
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error("LLM Chat API Error:", error);
      return {
        success: false,
        error:
          error instanceof Error ? error.message : "Failed to send chat query",
      };
    }
  }

  async testLLMConnection(): Promise<{ success: boolean; message: string }> {
    try {
      const response = await fetch(`${this.baseURL}/llm/test-connection`, {
        method: "GET",
        credentials: "include",
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error("LLM Connection Test Error:", error);
      return {
        success: false,
        message:
          error instanceof Error
            ? error.message
            : "Failed to test LLM connection",
      };
    }
  }

  // Helper method to prepare CRM data for the API
  prepareCRMData(crmContext: any): LLMChatRequest["crmData"] {
    return {
      leads: crmContext.leads || [],
      accounts: crmContext.accounts || [],
      contacts: crmContext.contacts || [],
      deals: crmContext.deals || [],
    };
  }

  // Helper method to format conversation history
  formatConversationHistory(
    messages: any[],
  ): Array<{ role: "user" | "assistant"; content: string }> {
    return messages
      .filter((msg) => msg.sender === "user" || msg.sender === "bot")
      .slice(-10) // Keep last 10 messages for context
      .map((msg) => ({
        role:
          msg.sender === "user" ? ("user" as const) : ("assistant" as const),
        content: msg.content,
      }));
  }
}

export const llmChatAPI = new LLMChatAPI();
