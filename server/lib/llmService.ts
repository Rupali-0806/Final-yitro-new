import OpenAI from 'openai';
import { 
  Contact, Account, ActivityLog, ActiveDeal, Lead 
} from '@shared/models';

interface CRMContext {
  leads: Lead[];
  accounts: Account[];
  contacts: Contact[];
  deals: ActiveDeal[];
  user?: {
    displayName?: string;
    email?: string;
    role?: string;
  };
}

interface LLMResponse {
  message: string;
  intent?: string;
  data?: any;
  quickActions?: string[];
}

export class LLMService {
  private openai: OpenAI;
  private isConfigured: boolean;

  constructor() {
    this.isConfigured = Boolean(process.env.OPENAI_API_KEY);
    
    if (this.isConfigured) {
      this.openai = new OpenAI({
        apiKey: process.env.OPENAI_API_KEY,
      });
    } else {
      console.warn('⚠️  OpenAI API key not configured. LLM features will be limited.');
    }
  }

  private createSystemPrompt(crmContext: CRMContext): string {
    const { leads, accounts, contacts, deals, user } = crmContext;
    
    return `You are an intelligent CRM assistant for ${user?.displayName || 'the user'}. You have access to their CRM data and should provide helpful, actionable insights.

Current CRM Data Overview:
- Leads: ${leads.length} total (${leads.filter(l => l.status === 'New').length} new, ${leads.filter(l => l.status === 'Qualified').length} qualified)
- Accounts: ${accounts.length} total (${accounts.filter(a => a.type === 'Customer').length} customers, ${accounts.filter(a => a.type === 'Prospect').length} prospects)
- Contacts: ${contacts.length} total
- Deals: ${deals.length} total (${deals.filter(d => !['Order Won', 'Order Lost'].includes(d.stage)).length} active)

Guidelines:
1. Always be helpful, professional, and action-oriented
2. Provide specific data when available rather than general statements
3. **IMPORTANT**: If user requests a specific number (e.g., "top 3 leads"), respond with EXACTLY that number - no more, no less
4. **PROVIDE PERSONALIZED RECOMMENDATIONS**: Always include specific "DO" and "DON'T" advice based on their actual CRM data
5. Suggest concrete next steps when appropriate
6. Keep responses clean and professional without emojis
7. Format responses in markdown for better readability
8. If asked about specific records, reference the actual data
9. Offer relevant quick actions for follow-up questions
10. Focus on insights that drive sales performance
11. **ANALYZE PATTERNS**: Look for concerning patterns in their data and provide warnings
12. **PRIORITIZE ACTIONS**: Help them focus on the most impactful activities

User Context:
- Name: ${user?.displayName || 'User'}
- Role: ${user?.role || 'Sales Professional'}

Respond naturally to user queries about their CRM data, sales performance, and provide actionable insights.`;
  }

  private createUserContext(query: string, crmContext: CRMContext): string {
    const { leads, accounts, contacts, deals } = crmContext;
    
    // Provide relevant data based on query keywords
    let contextData = '';
    const lowerQuery = query.toLowerCase();

    if (lowerQuery.includes('lead')) {
      // Extract number if specified (e.g., "top 3 leads", "5 best leads")
      const numberMatch = lowerQuery.match(/(?:top|best|first)\s*(\d+)|(\d+)\s*(?:top|best|leads)/);
      const requestedCount = numberMatch ? parseInt(numberMatch[1] || numberMatch[2]) : 5;
      const leadCount = Math.min(requestedCount, 10); // Cap at 10 for context efficiency

      const topLeads = leads
        .sort((a, b) => b.score - a.score)
        .slice(0, leadCount)
        .map(lead => `- ${lead.name} (${lead.company}): Score ${lead.score}, Value ${lead.value}, Status: ${lead.status}`)
        .join('\n');
      contextData += `\nTop ${leadCount} Leads (as requested):\n${topLeads}\n`;
    }

    if (lowerQuery.includes('deal') || lowerQuery.includes('pipeline')) {
      // Extract number if specified (e.g., "top 3 active deals", "5 best deals")
      const numberMatch = lowerQuery.match(/(?:top|best|first)\s*(\d+)|(\d+)\s*(?:top|best|active|deals)/);
      const requestedCount = numberMatch ? parseInt(numberMatch[1] || numberMatch[2]) : 5;
      const dealCount = Math.min(requestedCount, 10); // Cap at 10 for context efficiency

      const activeDeals = deals
        .filter(deal => !['Order Won', 'Order Lost'].includes(deal.stage))
        .sort((a, b) => b.dealValue - a.dealValue) // Sort by value for "top" deals
        .slice(0, dealCount)
        .map(deal => `- ${deal.dealName}: $${deal.dealValue.toLocaleString()}, Stage: ${deal.stage}, Closes: ${new Date(deal.closingDate).toLocaleDateString()}`)
        .join('\n');
      contextData += `\nTop ${dealCount} Active Deals (as requested):\n${activeDeals}\n`;
    }

    if (lowerQuery.includes('account') || lowerQuery.includes('customer')) {
      const topAccounts = accounts
        .filter(account => account.type === 'Customer')
        .sort((a, b) => b.activeDeals - a.activeDeals)
        .slice(0, 5)
        .map(account => `- ${account.name}: ${account.industry}, Revenue: ${account.revenue}, Active Deals: ${account.activeDeals}`)
        .join('\n');
      contextData += `\nTop Accounts:\n${topAccounts}\n`;
    }

    // Add personalized recommendations based on CRM data analysis
    const recommendations = this.generatePersonalizedRecommendations(crmContext);
    contextData += `\n${recommendations}`;

    return `User Query: "${query}"${contextData}`;
  }

  private generatePersonalizedRecommendations(crmContext: CRMContext): string {
    const { leads, accounts, contacts, deals } = crmContext;

    let recommendations = '\n📋 **PERSONALIZED INSIGHTS FOR YOUR RESPONSE:**\n';

    // Analyze leads performance
    const newLeads = leads.filter(l => l.status === 'New');
    const qualifiedLeads = leads.filter(l => l.status === 'Qualified');
    const workingLeads = leads.filter(l => l.status === 'Working');
    const lowScoreLeads = leads.filter(l => l.score < 30);
    const highScoreLeads = leads.filter(l => l.score >= 80);

    // Analyze deals performance
    const activeDeals = deals.filter(d => !['Order Won', 'Order Lost'].includes(d.stage));
    const highValueDeals = activeDeals.filter(d => d.dealValue > 50000);
    const lowProbabilityDeals = activeDeals.filter(d => d.probability < 30);
    const stallingDeals = activeDeals.filter(d => {
      const daysSinceUpdate = (Date.now() - new Date(d.lastActivity || d.createdAt).getTime()) / (1000 * 60 * 60 * 24);
      return daysSinceUpdate > 14;
    });
    const urgentDeals = activeDeals.filter(d => {
      const daysToClose = (new Date(d.closingDate).getTime() - Date.now()) / (1000 * 60 * 60 * 24);
      return daysToClose <= 7 && daysToClose > 0;
    });

    // Analyze accounts
    const customers = accounts.filter(a => a.type === 'Customer');
    const prospects = accounts.filter(a => a.type === 'Prospect');
    const inactiveAccounts = accounts.filter(a => a.activeDeals === 0);

    recommendations += '\n**WHAT TO DO (Priority Actions):**\n';

    // High-priority recommendations
    if (urgentDeals.length > 0) {
      recommendations += `- URGENT: ${urgentDeals.length} deals closing this week - focus on: ${urgentDeals[0].dealName}\n`;
    }

    if (highScoreLeads.length > 0) {
      recommendations += `- Contact your ${highScoreLeads.length} high-score leads (80+ score) immediately\n`;
    }

    if (highValueDeals.length > 0) {
      recommendations += `- Prioritize ${highValueDeals.length} high-value deals (>$50k) for maximum ROI\n`;
    }

    if (newLeads.length > 5) {
      recommendations += `- Qualify your ${newLeads.length} new leads within 24-48 hours\n`;
    }

    if (qualifiedLeads.length > 0) {
      recommendations += `- Convert ${qualifiedLeads.length} qualified leads to opportunities\n`;
    }

    if (inactiveAccounts.length > 0 && inactiveAccounts.length < accounts.length * 0.5) {
      recommendations += `- Re-engage ${Math.min(3, inactiveAccounts.length)} inactive accounts with new opportunities\n`;
    }

    recommendations += '\n**WHAT NOT TO DO (Avoid These):**\n';

    // Warning recommendations
    if (lowScoreLeads.length > leads.length * 0.3) {
      recommendations += `- Don't waste time on ${lowScoreLeads.length} low-score leads (<30) - focus on qualification first\n`;
    }

    if (stallingDeals.length > 0) {
      recommendations += `- Don't ignore ${stallingDeals.length} stalling deals - they need immediate attention\n`;
    }

    if (lowProbabilityDeals.length > 0) {
      recommendations += `- Don't over-invest in ${lowProbabilityDeals.length} low-probability deals (<30%) - reassess or disqualify\n`;
    }

    if (workingLeads.length > qualifiedLeads.length * 2) {
      recommendations += `- Don't let leads stagnate in "Working" status - move them forward or back to qualification\n`;
    }

    if (prospects.length > customers.length * 3) {
      recommendations += `- Don't chase too many prospects - focus on converting existing qualified ones first\n`;
    }

    // Performance insights
    const totalPipelineValue = activeDeals.reduce((sum, deal) => sum + deal.dealValue, 0);
    const avgDealValue = activeDeals.length > 0 ? totalPipelineValue / activeDeals.length : 0;
    const conversionRate = deals.filter(d => d.stage === 'Order Won').length / Math.max(deals.length, 1) * 100;

    recommendations += '\n**PERFORMANCE INSIGHTS:**\n';
    recommendations += `- Pipeline Value: $${totalPipelineValue.toLocaleString()}\n`;
    recommendations += `- Average Deal Size: $${Math.round(avgDealValue).toLocaleString()}\n`;
    recommendations += `- Conversion Rate: ${Math.round(conversionRate)}%\n`;

    if (conversionRate < 15) {
      recommendations += `- Focus on improving qualification process (conversion rate is low)\n`;
    }

    if (avgDealValue < 25000) {
      recommendations += `- Consider targeting higher-value opportunities\n`;
    }

    return recommendations;
  }

  async generateResponse(
    query: string, 
    crmContext: CRMContext, 
    conversationHistory: Array<{role: 'user' | 'assistant', content: string}> = []
  ): Promise<LLMResponse> {
    
    // Fallback if OpenAI is not configured
    if (!this.isConfigured) {
      return this.generateFallbackResponse(query, crmContext);
    }

    try {
      const systemPrompt = this.createSystemPrompt(crmContext);
      const userContext = this.createUserContext(query, crmContext);

      const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
        { role: 'system', content: systemPrompt },
        ...conversationHistory.slice(-6), // Keep last 6 messages for context
        { role: 'user', content: userContext }
      ];

      const completion = await this.openai.chat.completions.create({
        model: 'gpt-3.5-turbo',
        messages,
        max_tokens: 800,
        temperature: 0.7,
        presence_penalty: 0.1,
        frequency_penalty: 0.1,
      });

      const responseContent = completion.choices[0]?.message?.content || 'I apologize, but I couldn\'t generate a response at this time.';

      // Extract intent and suggest quick actions
      const intent = this.extractIntent(query);
      const quickActions = this.generateQuickActions(query, intent);

      return {
        message: responseContent,
        intent,
        quickActions
      };

    } catch (error) {
      console.error('LLM Service Error:', error);
      return this.generateFallbackResponse(query, crmContext);
    }
  }

  private generateFallbackResponse(query: string, crmContext: CRMContext): LLMResponse {
    const lowerQuery = query.toLowerCase();
    const { leads, accounts, contacts, deals } = crmContext;

    // Generate personalized recommendations for all responses
    const recommendations = this.generatePersonalizedRecommendations(crmContext);

    // Handle specific recommendation requests
    if (lowerQuery.includes('recommend') || lowerQuery.includes('advice') ||
        lowerQuery.includes('what should i') || lowerQuery.includes('what to do') ||
        lowerQuery.includes('help me prioritize') || lowerQuery.includes('suggestions')) {

      return {
        message: `🎯 **Personalized CRM Recommendations**\n\nBased on your current data analysis:${recommendations}\n\n💬 **Next Steps:** Ask me about specific areas you'd like to focus on!`,
        intent: 'recommendation_request',
        quickActions: ['Lead priorities', 'Deal strategy', 'Account planning']
      };
    }

    // Simple keyword-based fallback responses
    if (lowerQuery.includes('lead')) {
      // Extract specific number if requested
      const numberMatch = lowerQuery.match(/(?:top|give\s*me|show\s*me)\s*(\d+)|(\d+)\s*(?:top|best|leads)/);
      const requestedCount = numberMatch ? parseInt(numberMatch[1] || numberMatch[2]) : null;

      if (requestedCount) {
        const topLeads = leads
          .sort((a, b) => b.score - a.score)
          .slice(0, requestedCount);

        let response = `**Top ${requestedCount} Leads:**\n\n`;

        if (topLeads.length > 0) {
          topLeads.forEach((lead, index) => {
            response += `${index + 1}. **${lead.name}** from ${lead.company}\n`;
            response += `   Score: ${lead.score}/100 | Value: ${lead.value} | Status: ${lead.status}\n\n`;
          });
        } else {
          response += `No leads available to show.`;
        }

        return {
          message: response + recommendations,
          intent: 'lead_inquiry',
          quickActions: ['Lead details', 'Contact info', 'Next steps']
        };
      }

      const newLeads = leads.filter(l => l.status === 'New').length;
      const qualifiedLeads = leads.filter(l => l.status === 'Qualified').length;

      return {
        message: `📊 **Lead Summary**\n\nYou have ${leads.length} total leads:\n- ${newLeads} new leads\n- ${qualifiedLeads} qualified leads\n\nWould you like me to show you the top performing leads or help you prioritize your outreach?${recommendations}`,
        intent: 'lead_inquiry',
        quickActions: ['Show top leads', 'Lead priorities', 'This week\'s leads']
      };
    }

    if (lowerQuery.includes('deal') || lowerQuery.includes('pipeline')) {
      // Extract specific number if requested
      const numberMatch = lowerQuery.match(/(?:top|give\s*me|show\s*me)\s*(\d+)|(\d+)\s*(?:top|best|active|deals)/);
      const requestedCount = numberMatch ? parseInt(numberMatch[1] || numberMatch[2]) : null;

      const activeDeals = deals.filter(d => !['Order Won', 'Order Lost'].includes(d.stage));

      if (requestedCount && activeDeals.length > 0) {
        const topDeals = activeDeals
          .sort((a, b) => b.dealValue - a.dealValue)
          .slice(0, requestedCount);

        let response = `💼 **Top ${requestedCount} Active Deals:**\n\n`;

        topDeals.forEach((deal, index) => {
          response += `${index + 1}. **${deal.dealName}**\n`;
          response += `   💰 Value: $${deal.dealValue.toLocaleString()} | 🔄 Stage: ${deal.stage}\n`;
          response += `   📅 Closes: ${new Date(deal.closingDate).toLocaleDateString()} | 📈 ${deal.probability}% likely\n\n`;
        });

        const totalValue = topDeals.reduce((sum, deal) => sum + deal.dealValue, 0);
        response += `💎 **Total Value**: $${totalValue.toLocaleString()}`;

        return {
          message: response + recommendations,
          intent: 'deal_inquiry',
          quickActions: ['Deal details', 'Next steps', 'Pipeline analysis']
        };
      }

      const totalValue = activeDeals.reduce((sum, deal) => sum + deal.dealValue, 0);

      return {
        message: `💼 **Pipeline Overview**\n\nYou have ${activeDeals.length} active deals worth $${totalValue.toLocaleString()} total.\n\nWould you like to see deals closing soon or need help prioritizing your pipeline?${recommendations}`,
        intent: 'deal_inquiry',
        quickActions: ['Top active deals', 'Deals closing soon', 'High value deals']
      };
    }

    if (lowerQuery.includes('account') || lowerQuery.includes('customer')) {
      const customers = accounts.filter(a => a.type === 'Customer').length;
      const prospects = accounts.filter(a => a.type === 'Prospect').length;
      
      return {
        message: `🏢 **Account Overview**\n\nYou're managing ${accounts.length} total accounts:\n- ${customers} customers\n- ${prospects} prospects\n\nWhat would you like to know about your accounts?${recommendations}`,
        intent: 'account_inquiry',
        quickActions: ['Top customers', 'New prospects', 'Account health']
      };
    }

    // General response
    return {
      message: `I understand you're asking about "${query}". I can help you with leads, deals, accounts, and contacts. Try asking me something like:\n\n• "Show me my top leads"\n• "What deals are closing soon?"\n• "Account performance summary"\n• "Contact activity this week"${recommendations}`,
      intent: 'general_inquiry',
      quickActions: ['Top leads', 'Deals closing soon', 'Account summary', 'Recent activity']
    };
  }

  private extractIntent(query: string): string {
    const lowerQuery = query.toLowerCase();
    
    if (lowerQuery.includes('lead')) return 'lead_inquiry';
    if (lowerQuery.includes('deal') || lowerQuery.includes('pipeline')) return 'deal_inquiry';
    if (lowerQuery.includes('account') || lowerQuery.includes('customer')) return 'account_inquiry';
    if (lowerQuery.includes('contact')) return 'contact_inquiry';
    if (lowerQuery.includes('performance') || lowerQuery.includes('metric')) return 'performance_inquiry';
    if (lowerQuery.includes('help') || lowerQuery.includes('how')) return 'help_request';
    
    return 'general_inquiry';
  }

  private generateQuickActions(query: string, intent: string): string[] {
    const baseActions: Record<string, string[]> = {
      lead_inquiry: ['Show top leads', 'New leads this week', 'Lead priorities'],
      deal_inquiry: ['Pipeline analysis', 'Deals closing soon', 'Won/lost deals'],
      account_inquiry: ['Top customers', 'Account health', 'New opportunities'],
      contact_inquiry: ['Recent contacts', 'Contact activity', 'Follow-ups needed'],
      performance_inquiry: ['Monthly metrics', 'Goal progress', 'Team comparison'],
      help_request: ['Getting started', 'Common questions', 'Feature tour'],
      general_inquiry: ['Top leads', 'Pipeline status', 'Account summary']
    };

    return baseActions[intent] || baseActions.general_inquiry;
  }

  // Test if the service is properly configured
  async testConnection(): Promise<{ success: boolean; message: string }> {
    if (!this.isConfigured) {
      return {
        success: false,
        message: 'OpenAI API key not configured. Set OPENAI_API_KEY environment variable.'
      };
    }

    try {
      await this.openai.chat.completions.create({
        model: 'gpt-3.5-turbo',
        messages: [{ role: 'user', content: 'Test connection' }],
        max_tokens: 10
      });

      return {
        success: true,
        message: 'LLM service connected successfully'
      };
    } catch (error: any) {
      return {
        success: false,
        message: `LLM connection failed: ${error.message}`
      };
    }
  }
}

export const llmService = new LLMService();
