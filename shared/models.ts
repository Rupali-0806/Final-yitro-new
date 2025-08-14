// Database Models for CRM Platform

export interface Contact {
  id: string;
  firstName: string;
  lastName: string;
  title?: string;
  associatedAccount?: string;
  emailAddress?: string;
  deskPhone?: string;
  mobilePhone?: string;
  city?: string;
  state?: string;
  country?: string;
  timeZone?: string;
  source?: 'Data Research' | 'Referral' | 'Event';
  owner?: string;
  status?: 'Suspect' | 'Prospect' | 'Active Deal' | 'Do Not Call';
  createdAt: Date;
  updatedAt: Date;
  createdBy: string;
  updatedBy: string;
}

export interface Account {
  id: string;
  accountName: string;
  accountRating?: 'Platinum (Must Have)' | 'Gold (High Priority)' | 'Silver (Medium Priority)' | 'Bronze (Low Priority)';
  accountOwner?: string;
  status?: 'Suspect' | 'Prospect' | 'Active Deal' | 'Do Not Call';
  industry?: string;
  revenue?: string;
  numberOfEmployees?: string;
  addressLine1?: string;
  addressLine2?: string;
  city?: string;
  state?: string;
  country?: string;
  zipPostCode?: string;
  timeZone?: string;
  boardNumber?: string;
  website?: string;
  geo?: 'Americas' | 'India' | 'Philippines' | 'EMEA' | 'ANZ';
  createdAt: Date;
  updatedAt: Date;
  createdBy: string;
  updatedBy: string;
}

export interface ActivityLog {
  id: string;
  activityType: 'Call' | 'Email' | 'LinkedIn Msg' | 'SMS' | 'Other';
  associatedContact?: string;
  associatedAccount?: string;
  dateTime: Date;
  followUpSchedule?: string;
  summary?: string;
  outcomeDisposition?: 'Voicemail' | 'RNR' | 'Meeting Fixed' | 'Meeting Completed' | 'Meeting Rescheduled' | 'Not Interested' | 'Do not Call' | 'Callback requested' | 'Email sent' | 'Email Received';
  createdAt: Date;
  updatedAt: Date;
  createdBy: string;
  updatedBy: string;
}

export interface ActiveDeal {
  id: string;
  dealOwner?: string;
  dealName: string;
  businessLine?: 'Human Capital' | 'Managed Services' | 'GCC' | 'Automation' | 'Support' | 'Product' | 'Solution' | 'RCM';
  associatedAccount?: string;
  associatedContact?: string;
  closingDate?: Date;
  probability?: '30-100%';
  dealValue?: string;
  approvedBy?: string;
  description?: string;
  nextStep?: string;
  geo?: 'Americas' | 'India' | 'Philippines' | 'EMEA' | 'ANZ';
  entity?: 'Yitro Global' | 'Yitro Tech';
  stage?: 'Opportunity Identified' | 'Proposal Submitted' | 'Negotiating' | 'Closing' | 'Order Won' | 'Order Lost';
  createdAt: Date;
  updatedAt: Date;
  createdBy: string;
  updatedBy: string;
}

export interface Lead {
  id: string;
  firstName: string;
  lastName: string;
  company: string;
  title?: string;
  phone?: string;
  email?: string;
  leadSource?: 'Website' | 'Referral' | 'Trade Show' | 'Cold Call' | 'Email' | 'Partner';
  status?: 'New' | 'Working' | 'Qualified' | 'Unqualified';
  rating?: 'Hot' | 'Warm' | 'Cold';
  owner?: string;
  createdAt: Date;
  updatedAt: Date;
  createdBy: string;
  updatedBy: string;
}

export interface ReportData {
  id: string;
  reportType: 'sales-performance' | 'activity-summary' | 'pipeline-analysis';
  period: 'month' | 'quarter' | 'year';
  salesRep?: string;
  geo?: 'Americas' | 'India' | 'Philippines' | 'EMEA' | 'ANZ';
  businessLine?: string;
  org?: string;
  metrics: {
    meetingsFixed?: number;
    meetingsCompleted?: number;
    opportunityCreatedNos?: number;
    opportunityCreatedValue?: number;
    proposalSubmittedNos?: number;
    proposalSubmittedValue?: number;
    orderWonNos?: number;
    orderWonValue?: number;
    orderLostNos?: number;
    orderLostValue?: number;
    // vs Target metrics
    meetingsFixedVsTarget?: number;
    meetingsCompletedVsTarget?: number;
    opportunityCreatedVsTarget?: number;
    opportunityCreatedValueVsTarget?: number;
    proposalSubmittedVsTarget?: number;
    proposalSubmittedValueVsTarget?: number;
    orderWonVsTarget?: number;
    orderWonValueVsTarget?: number;
  };
  generatedAt: Date;
  generatedBy: string;
}

// API Response Types
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  message?: string;
  error?: string;
}

export interface PaginatedResponse<T> {
  success: boolean;
  data: T[];
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
  };
}

// Request Types
export interface CreateContactRequest extends Omit<Contact, 'id' | 'createdAt' | 'updatedAt' | 'createdBy' | 'updatedBy'> {}
export interface UpdateContactRequest extends Partial<CreateContactRequest> {}

export interface CreateAccountRequest extends Omit<Account, 'id' | 'createdAt' | 'updatedAt' | 'createdBy' | 'updatedBy'> {}
export interface UpdateAccountRequest extends Partial<CreateAccountRequest> {}

export interface CreateActivityRequest extends Omit<ActivityLog, 'id' | 'createdAt' | 'updatedAt' | 'createdBy' | 'updatedBy'> {}
export interface UpdateActivityRequest extends Partial<CreateActivityRequest> {}

export interface CreateDealRequest extends Omit<ActiveDeal, 'id' | 'createdAt' | 'updatedAt' | 'createdBy' | 'updatedBy'> {}
export interface UpdateDealRequest extends Partial<CreateDealRequest> {}

export interface CreateLeadRequest extends Omit<Lead, 'id' | 'createdAt' | 'updatedAt' | 'createdBy' | 'updatedBy'> {}
export interface UpdateLeadRequest extends Partial<CreateLeadRequest> {}

export interface ReportRequest {
  reportType: 'sales-performance' | 'activity-summary' | 'pipeline-analysis';
  period: 'month' | 'quarter' | 'year';
  startDate?: Date;
  endDate?: Date;
  salesRep?: string;
  geo?: string;
  businessLine?: string;
  org?: string;
  format?: 'json' | 'excel' | 'pdf';
}

export interface UserProfile {
  id: string;
  firstName: string;
  lastName: string;
  email: string;
  phone?: string;
  title?: string;
  department?: string;
  role?: 'Admin' | 'Sales Manager' | 'Sales Rep' | 'User';
  profilePhoto?: string;
  timezone?: string;
  language?: string;
  notifications?: {
    emailNotifications: boolean;
    smsNotifications: boolean;
    pushNotifications: boolean;
  };
  createdAt: Date;
  updatedAt: Date;
}

export interface CreateUserProfileRequest extends Omit<UserProfile, 'id' | 'createdAt' | 'updatedAt'> {}
export interface UpdateUserProfileRequest extends Partial<CreateUserProfileRequest> {}
