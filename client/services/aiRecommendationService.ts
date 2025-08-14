import { ActiveDeal } from "../../shared/models";

// Stage mapping from the XGBoost model
const STAGE_MAPPING = {
  'Negotiating': 0,
  'Proposal Submitted': 1,
  'Closing': 2,
  'Order Won': 3,
  'Opportunity Identified': 4,
  'Opportunity Ident.': 4 // Handle abbreviated form
} as const;

// Action mapping from the XGBoost model
const ACTION_MAPPING = {
  0: 'Send follow-up email',
  1: 'Make a call',
  2: 'Send proposal',
  3: 'Schedule a meeting/demo',
  4: 'Share a case study/testimonial',
  5: 'Offer a discount',
  6: 'Do nothing (wait)'
} as const;

// Action priority mapping for display
const ACTION_PRIORITIES = {
  0: 'medium', // Send follow-up email
  1: 'high',   // Make a call
  2: 'high',   // Send proposal
  3: 'high',   // Schedule a meeting/demo
  4: 'medium', // Share a case study/testimonial
  5: 'medium', // Offer a discount
  6: 'low'     // Do nothing (wait)
} as const;

export interface AIRecommendation {
  id: string;
  dealId: string;
  dealName: string;
  action: string;
  confidence: number;
  priority: 'high' | 'medium' | 'low';
  reason: string;
  actionType: 'call' | 'email' | 'meeting' | 'proposal' | 'case-study' | 'discount' | 'wait';
}

class AIRecommendationService {
  // Simplified decision tree implementation based on the XGBoost model training data
  private predictAction(value: number, probability: number, stageEncoded: number): { action: number; confidence: number } {
    // High-level decision tree logic based on the patterns in the training data
    
    // High probability deals (>= 80%)
    if (probability >= 80) {
      if (stageEncoded === 2) { // Closing
        return { action: 3, confidence: 0.92 }; // Schedule meeting/demo
      }
      if (stageEncoded === 0) { // Negotiating
        return { action: 2, confidence: 0.88 }; // Send proposal/contract
      }
      if (stageEncoded === 1) { // Proposal Submitted
        return { action: 3, confidence: 0.85 }; // Schedule meeting/demo
      }
    }
    
    // Medium probability deals (50-79%)
    if (probability >= 50 && probability < 80) {
      if (stageEncoded === 1) { // Proposal Submitted
        if (value >= 150000) {
          return { action: 3, confidence: 0.78 }; // Schedule meeting/demo for high value
        } else {
          return { action: 0, confidence: 0.72 }; // Follow up email
        }
      }
      if (stageEncoded === 0) { // Negotiating
        return { action: 3, confidence: 0.75 }; // Schedule meeting/demo
      }
      if (stageEncoded === 4) { // Opportunity Identified
        return { action: 0, confidence: 0.70 }; // Follow up email
      }
    }
    
    // Low probability deals (< 50%)
    if (probability < 50) {
      if (stageEncoded === 4) { // Opportunity Identified
        if (value >= 100000) {
          return { action: 1, confidence: 0.65 }; // Make a call for high value
        } else {
          return { action: 4, confidence: 0.62 }; // Send case study
        }
      }
      if (stageEncoded === 1) { // Proposal Submitted
        return { action: 0, confidence: 0.68 }; // Follow up email
      }
      if (stageEncoded === 0) { // Negotiating
        return { action: 0, confidence: 0.66 }; // Follow up email
      }
    }
    
    // Default case
    return { action: 0, confidence: 0.55 }; // Follow up email
  }

  private getActionType(actionId: number): AIRecommendation['actionType'] {
    switch (actionId) {
      case 0: return 'email';
      case 1: return 'call';
      case 2: return 'proposal';
      case 3: return 'meeting';
      case 4: return 'case-study';
      case 5: return 'discount';
      case 6: return 'wait';
      default: return 'email';
    }
  }

  private generateReason(deal: ActiveDeal, actionId: number, confidence: number): string {
    const value = typeof deal.dealValue === 'string' 
      ? parseFloat(deal.dealValue.replace(/[^0-9.-]+/g, "")) 
      : deal.dealValue || 0;
    
    const probability = deal.probability || 0;
    const stage = deal.stage || 'Unknown';

    switch (actionId) {
      case 0: // Send follow-up email
        if (probability < 50) {
          return `Low probability (${probability}%) deal in ${stage} stage needs gentle follow-up to maintain engagement.`;
        }
        return `Medium probability (${probability}%) deal requires follow-up to move forward in the pipeline.`;
      
      case 1: // Make a call
        return `High-value deal ($${value.toLocaleString()}) with ${probability}% probability needs direct phone contact for maximum impact.`;
      
      case 2: // Send proposal
        if (stage === 'Negotiating') {
          return `Deal in negotiation stage with ${probability}% probability is ready for contract/proposal submission.`;
        }
        return `High probability (${probability}%) deal in ${stage} stage is ready for proposal advancement.`;
      
      case 3: // Schedule meeting/demo
        if (probability >= 80) {
          return `High probability (${probability}%) deal needs face-to-face meeting to close successfully.`;
        }
        return `Deal with ${probability}% probability in ${stage} stage benefits from direct meeting/demo.`;
      
      case 4: // Share case study
        return `Deal with ${probability}% probability needs social proof through case studies to build confidence.`;
      
      case 5: // Offer discount
        return `Deal in ${stage} stage with ${probability}% probability may benefit from pricing incentives.`;
      
      case 6: // Do nothing
        return `Deal appears to be progressing well on its own - monitor without immediate action.`;
      
      default:
        return `AI analysis suggests this action based on deal characteristics and historical patterns.`;
    }
  }

  public generateRecommendations(deals: ActiveDeal[]): AIRecommendation[] {
    return deals
      .filter(deal => !['Order Won', 'Order Lost'].includes(deal.stage || ''))
      .map(deal => {
        const value = typeof deal.dealValue === 'string' 
          ? parseFloat(deal.dealValue.replace(/[^0-9.-]+/g, "")) 
          : deal.dealValue || 0;
        
        const probability = deal.probability || 0;
        const stageEncoded = STAGE_MAPPING[deal.stage as keyof typeof STAGE_MAPPING] ?? 4;

        const prediction = this.predictAction(value, probability, stageEncoded);
        const actionText = ACTION_MAPPING[prediction.action as keyof typeof ACTION_MAPPING];
        const priority = ACTION_PRIORITIES[prediction.action as keyof typeof ACTION_PRIORITIES];
        const actionType = this.getActionType(prediction.action);
        const reason = this.generateReason(deal, prediction.action, prediction.confidence);

        return {
          id: `ai-rec-${deal.id}`,
          dealId: deal.id,
          dealName: deal.dealName,
          action: actionText,
          confidence: prediction.confidence,
          priority,
          reason,
          actionType
        };
      })
      .sort((a, b) => {
        // Sort by priority first, then by confidence
        const priorityOrder = { high: 3, medium: 2, low: 1 };
        const priorityDiff = priorityOrder[b.priority] - priorityOrder[a.priority];
        if (priorityDiff !== 0) return priorityDiff;
        return b.confidence - a.confidence;
      })
      .slice(0, 10); // Limit to top 10 recommendations
  }

  public getRecommendationForDeal(deal: ActiveDeal): AIRecommendation | null {
    const recommendations = this.generateRecommendations([deal]);
    return recommendations.length > 0 ? recommendations[0] : null;
  }

  // Get recommendations by priority
  public getHighPriorityRecommendations(deals: ActiveDeal[]): AIRecommendation[] {
    return this.generateRecommendations(deals).filter(rec => rec.priority === 'high');
  }

  // Get recommendations by action type
  public getRecommendationsByActionType(deals: ActiveDeal[], actionType: AIRecommendation['actionType']): AIRecommendation[] {
    return this.generateRecommendations(deals).filter(rec => rec.actionType === actionType);
  }
}

export const aiRecommendationService = new AIRecommendationService();
