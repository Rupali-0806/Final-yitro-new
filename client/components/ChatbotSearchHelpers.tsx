import { Lead, Account, Contact, ActiveDeal } from "../contexts/CRMContext";

export interface SearchResult {
  type: "lead" | "account" | "contact" | "deal";
  data: any;
  relevanceScore: number;
}

export class ChatbotSearchEngine {
  static searchContacts(contacts: Contact[], query: string): SearchResult[] {
    const searchTerms = query.toLowerCase().split(" ");

    return contacts
      .map((contact) => {
        let score = 0;
        const searchableText =
          `${contact.firstName} ${contact.lastName} ${contact.emailAddress} ${contact.associatedAccount} ${contact.title}`.toLowerCase();

        searchTerms.forEach((term) => {
          if (searchableText.includes(term)) {
            score += 1;
            // Boost score for name matches
            if (
              `${contact.firstName} ${contact.lastName}`
                .toLowerCase()
                .includes(term)
            ) {
              score += 2;
            }
            // Boost score for exact company matches
            if (contact.associatedAccount?.toLowerCase().includes(term)) {
              score += 1.5;
            }
          }
        });

        return {
          type: "contact" as const,
          data: contact,
          relevanceScore: score,
        };
      })
      .filter((result) => result.relevanceScore > 0)
      .sort((a, b) => b.relevanceScore - a.relevanceScore);
  }

  static searchAccounts(accounts: Account[], query: string): SearchResult[] {
    const searchTerms = query.toLowerCase().split(" ");

    return accounts
      .map((account) => {
        let score = 0;
        const searchableText =
          `${account.name} ${account.industry} ${account.type} ${account.location}`.toLowerCase();

        searchTerms.forEach((term) => {
          if (searchableText.includes(term)) {
            score += 1;
            // Boost score for name matches
            if (account.name.toLowerCase().includes(term)) {
              score += 2;
            }
          }
        });

        return {
          type: "account" as const,
          data: account,
          relevanceScore: score,
        };
      })
      .filter((result) => result.relevanceScore > 0)
      .sort((a, b) => b.relevanceScore - a.relevanceScore);
  }

  static searchLeads(leads: Lead[], query: string): SearchResult[] {
    const searchTerms = query.toLowerCase().split(" ");

    return leads
      .map((lead) => {
        let score = 0;
        const searchableText =
          `${lead.name} ${lead.company} ${lead.title} ${lead.email} ${lead.source}`.toLowerCase();

        searchTerms.forEach((term) => {
          if (searchableText.includes(term)) {
            score += 1;
            // Boost score for name matches
            if (lead.name.toLowerCase().includes(term)) {
              score += 2;
            }
            // Boost score for company matches
            if (lead.company.toLowerCase().includes(term)) {
              score += 1.5;
            }
          }
        });

        return {
          type: "lead" as const,
          data: lead,
          relevanceScore: score,
        };
      })
      .filter((result) => result.relevanceScore > 0)
      .sort((a, b) => b.relevanceScore - a.relevanceScore);
  }

  static searchDeals(deals: ActiveDeal[], query: string): SearchResult[] {
    const searchTerms = query.toLowerCase().split(" ");

    return deals
      .map((deal) => {
        let score = 0;
        const searchableText =
          `${deal.dealName} ${deal.associatedAccount} ${deal.associatedContact} ${deal.businessLine} ${deal.stage}`.toLowerCase();

        searchTerms.forEach((term) => {
          if (searchableText.includes(term)) {
            score += 1;
            // Boost score for deal name matches
            if (deal.dealName.toLowerCase().includes(term)) {
              score += 2;
            }
            // Boost score for account matches
            if (deal.associatedAccount.toLowerCase().includes(term)) {
              score += 1.5;
            }
          }
        });

        return {
          type: "deal" as const,
          data: deal,
          relevanceScore: score,
        };
      })
      .filter((result) => result.relevanceScore > 0)
      .sort((a, b) => b.relevanceScore - a.relevanceScore);
  }

  static performGlobalSearch(
    query: string,
    leads: Lead[],
    accounts: Account[],
    contacts: Contact[],
    deals: ActiveDeal[],
  ): SearchResult[] {
    const leadResults = this.searchLeads(leads, query);
    const accountResults = this.searchAccounts(accounts, query);
    const contactResults = this.searchContacts(contacts, query);
    const dealResults = this.searchDeals(deals, query);

    return [
      ...leadResults,
      ...accountResults,
      ...contactResults,
      ...dealResults,
    ]
      .sort((a, b) => b.relevanceScore - a.relevanceScore)
      .slice(0, 10); // Limit to top 10 results
  }

  static formatSearchResults(results: SearchResult[]): string {
    if (results.length === 0) {
      return "🔍 No results found. Try searching with different terms or check your spelling.";
    }

    let response = `🔍 **Search Results (${results.length} found):**\n\n`;

    results.forEach((result, index) => {
      const data = result.data;

      switch (result.type) {
        case "contact":
          response += `👤 ${index + 1}. **${data.firstName} ${data.lastName}**\n`;
          response += `   📧 ${data.emailAddress || "No email"}\n`;
          response += `   🏢 ${data.associatedAccount || "No account"}\n`;
          response += `   💼 ${data.title || "No title"}\n`;
          response += `   📱 ${data.mobilePhone || data.deskPhone || "No phone"}\n`;
          break;

        case "account":
          response += `🏢 ${index + 1}. **${data.name}**\n`;
          response += `   🏭 Industry: ${data.industry}\n`;
          response += `   💰 Revenue: ${data.revenue}\n`;
          response += `   📍 Location: ${data.location}\n`;
          response += `   🎯 Type: ${data.type}\n`;
          break;

        case "lead":
          response += `🎯 ${index + 1}. **${data.name}** from ${data.company}\n`;
          response += `   📊 Score: ${data.score}/100\n`;
          response += `   💰 Value: ${data.value}\n`;
          response += `   📧 Email: ${data.email}\n`;
          response += `   🔥 Status: ${data.status}\n`;
          break;

        case "deal":
          response += `💼 ${index + 1}. **${data.dealName}**\n`;
          response += `   🏢 Account: ${data.associatedAccount}\n`;
          response += `   💰 Value: $${data.dealValue.toLocaleString()}\n`;
          response += `   📅 Closing: ${new Date(data.closingDate).toLocaleDateString()}\n`;
          response += `   🔄 Stage: ${data.stage}\n`;
          break;
      }

      response += "\n";
    });

    return response.trim();
  }

  static extractSearchQuery(input: string): string | null {
    const searchPatterns = [
      /find (.+)/i,
      /search (?:for )?(.+)/i,
      /show me (.+)/i,
      /look for (.+)/i,
      /get (.+)/i,
    ];

    for (const pattern of searchPatterns) {
      const match = input.match(pattern);
      if (match) {
        return match[1].trim();
      }
    }

    return null;
  }
}
