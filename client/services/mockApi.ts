// Mock API for frontend demo deployment
import { 
  Contact, Account, ActivityLog, ActiveDeal, Lead, UserProfile,
  ApiResponse, PaginatedResponse
} from '@shared/models';

// Mock data for demo
const mockContacts: Contact[] = [
  {
    id: '1',
    firstName: 'John',
    lastName: 'Smith',
    title: 'CEO',
    associatedAccount: 'TechCorp Solutions',
    emailAddress: 'john.smith@techcorp.com',
    deskPhone: '+1-555-0123',
    mobilePhone: '+1-555-0124',
    city: 'New York',
    state: 'NY',
    country: 'USA',
    timeZone: 'EST',
    source: 'Data Research',
    owner: 'Sales Rep 1',
    status: 'Prospect',
    createdAt: new Date('2024-01-01'),
    updatedAt: new Date('2024-01-07'),
    createdBy: 'system',
    updatedBy: 'system'
  },
  {
    id: '2',
    firstName: 'Jane',
    lastName: 'Doe',
    title: 'CTO',
    associatedAccount: 'Innovate Inc',
    emailAddress: 'jane.doe@innovate.com',
    deskPhone: '+1-555-0125',
    mobilePhone: '+1-555-0126',
    city: 'San Francisco',
    state: 'CA',
    country: 'USA',
    timeZone: 'PST',
    source: 'Referral',
    owner: 'Sales Rep 2',
    status: 'Active Deal',
    createdAt: new Date('2024-01-02'),
    updatedAt: new Date('2024-01-07'),
    createdBy: 'system',
    updatedBy: 'system'
  }
];

const mockAccounts: Account[] = [
  {
    id: '1',
    accountName: 'TechCorp Solutions',
    accountRating: 'Gold (High Priority)',
    accountOwner: 'Sales Rep 1',
    status: 'Prospect',
    industry: 'Technology',
    revenue: '$10M - $50M',
    numberOfEmployees: '100-500',
    addressLine1: '123 Tech Street',
    city: 'New York',
    state: 'NY',
    country: 'USA',
    zipPostCode: '10001',
    timeZone: 'EST',
    website: 'https://techcorp.com',
    geo: 'Americas',
    createdAt: new Date('2024-01-01'),
    updatedAt: new Date('2024-01-07'),
    createdBy: 'system',
    updatedBy: 'system'
  },
  {
    id: '2',
    accountName: 'Innovate Inc',
    accountRating: 'Platinum (Must Have)',
    accountOwner: 'Sales Rep 2',
    status: 'Active Deal',
    industry: 'Software',
    revenue: '$50M+',
    numberOfEmployees: '500+',
    addressLine1: '456 Innovation Ave',
    city: 'San Francisco',
    state: 'CA',
    country: 'USA',
    zipPostCode: '94105',
    timeZone: 'PST',
    website: 'https://innovate.com',
    geo: 'Americas',
    createdAt: new Date('2024-01-02'),
    updatedAt: new Date('2024-01-07'),
    createdBy: 'system',
    updatedBy: 'system'
  }
];

const mockDeals: ActiveDeal[] = [
  {
    id: '1',
    dealOwner: 'Sales Rep 1',
    dealName: 'TechCorp Automation Project',
    businessLine: 'Automation',
    associatedAccount: 'TechCorp Solutions',
    associatedContact: 'John Smith',
    closingDate: new Date('2024-03-15'),
    probability: '75%',
    dealValue: '$150,000',
    approvedBy: 'Sales Manager',
    description: 'Large automation project for TechCorp',
    nextStep: 'Schedule technical review',
    geo: 'Americas',
    entity: 'Yitro Global',
    stage: 'Proposal Submitted',
    createdAt: new Date('2024-01-01'),
    updatedAt: new Date('2024-01-07'),
    createdBy: 'system',
    updatedBy: 'system'
  },
  {
    id: '2',
    dealOwner: 'Sales Rep 2',
    dealName: 'Innovate Software Development',
    businessLine: 'Product',
    associatedAccount: 'Innovate Inc',
    associatedContact: 'Jane Doe',
    closingDate: new Date('2024-02-28'),
    probability: '90%',
    dealValue: '$250,000',
    approvedBy: 'Sales Manager',
    description: 'Custom software development for Innovate Inc',
    nextStep: 'Final contract review',
    geo: 'Americas',
    entity: 'Yitro Tech',
    stage: 'Negotiating',
    createdAt: new Date('2024-01-02'),
    updatedAt: new Date('2024-01-07'),
    createdBy: 'system',
    updatedBy: 'system'
  },
  {
    id: '3',
    dealOwner: 'Sales Rep 1',
    dealName: 'Enterprise Analytics Platform',
    businessLine: 'Solution',
    associatedAccount: 'Enterprise Corp',
    associatedContact: 'Mike Johnson',
    closingDate: new Date('2024-01-15'),
    probability: '100%',
    dealValue: '$500,000',
    approvedBy: 'Sales Director',
    description: 'Complete analytics solution for large enterprise',
    nextStep: 'Implementation kickoff',
    geo: 'Americas',
    entity: 'Yitro Global',
    stage: 'Order Won',
    createdAt: new Date('2023-12-01'),
    updatedAt: new Date('2024-01-15'),
    createdBy: 'system',
    updatedBy: 'system'
  }
];

const mockLeads: Lead[] = [
  {
    id: '1',
    firstName: 'Alice',
    lastName: 'Johnson',
    company: 'StartupCorp',
    title: 'Founder',
    phone: '+1-555-0200',
    email: 'alice@startupcorp.com',
    leadSource: 'Website',
    status: 'New',
    rating: 'Hot',
    owner: 'Sales Rep 1',
    createdAt: new Date('2024-01-05'),
    updatedAt: new Date('2024-01-07'),
    createdBy: 'system',
    updatedBy: 'system'
  },
  {
    id: '2',
    firstName: 'Bob',
    lastName: 'Wilson',
    company: 'Enterprise Ltd',
    title: 'CTO',
    phone: '+1-555-0201',
    email: 'bob@enterprise.com',
    leadSource: 'Referral',
    status: 'Working',
    rating: 'Warm',
    owner: 'Sales Rep 2',
    createdAt: new Date('2024-01-06'),
    updatedAt: new Date('2024-01-07'),
    createdBy: 'system',
    updatedBy: 'system'
  }
];

const mockActivities: ActivityLog[] = [
  {
    id: '1',
    activityType: 'Call',
    associatedContact: 'John Smith',
    associatedAccount: 'TechCorp Solutions',
    dateTime: new Date('2024-01-07T10:00:00Z'),
    followUpSchedule: '2024-01-14T10:00:00Z',
    summary: 'Initial discovery call with TechCorp. Discussed automation needs and project scope.',
    outcomeDisposition: 'Meeting Completed',
    createdAt: new Date('2024-01-07'),
    updatedAt: new Date('2024-01-07'),
    createdBy: 'system',
    updatedBy: 'system'
  },
  {
    id: '2',
    activityType: 'Email',
    associatedContact: 'Jane Doe',
    associatedAccount: 'Innovate Inc',
    dateTime: new Date('2024-01-07T14:30:00Z'),
    followUpSchedule: '',
    summary: 'Sent proposal document and technical specifications to Innovate Inc.',
    outcomeDisposition: 'Email Sent',
    createdAt: new Date('2024-01-07'),
    updatedAt: new Date('2024-01-07'),
    createdBy: 'system',
    updatedBy: 'system'
  }
];

const mockUserProfile: UserProfile = {
  id: '1',
  firstName: 'Demo',
  lastName: 'User',
  email: 'demo@yitro.com',
  phone: '+1-555-0100',
  title: 'Sales Manager',
  department: 'Sales',
  role: 'Sales Manager',
  profilePhoto: '',
  timezone: 'EST',
  language: 'en',
  notifications: {
    emailNotifications: true,
    smsNotifications: false,
    pushNotifications: true
  },
  createdAt: new Date('2024-01-01'),
  updatedAt: new Date('2024-01-07')
};

// Mock API implementation
const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

export const mockApi = {
  contacts: {
    async getAll(): Promise<PaginatedResponse<Contact>> {
      await delay(300); // Simulate network delay
      return {
        success: true,
        data: mockContacts,
        pagination: {
          page: 1,
          limit: 10,
          total: mockContacts.length,
          totalPages: 1
        }
      };
    },

    async getById(id: string): Promise<ApiResponse<Contact>> {
      await delay(200);
      const contact = mockContacts.find(c => c.id === id);
      return {
        success: !!contact,
        data: contact,
        message: contact ? 'Contact found' : 'Contact not found'
      };
    },

    async create(data: any): Promise<ApiResponse<Contact>> {
      await delay(400);
      const newContact: Contact = {
        id: String(mockContacts.length + 1),
        ...data,
        createdAt: new Date(),
        updatedAt: new Date(),
        createdBy: 'demo-user',
        updatedBy: 'demo-user'
      };
      mockContacts.push(newContact);
      return {
        success: true,
        data: newContact,
        message: 'Contact created successfully'
      };
    },

    async update(id: string, data: any): Promise<ApiResponse<Contact>> {
      await delay(400);
      const index = mockContacts.findIndex(c => c.id === id);
      if (index !== -1) {
        mockContacts[index] = { ...mockContacts[index], ...data, updatedAt: new Date() };
        return {
          success: true,
          data: mockContacts[index],
          message: 'Contact updated successfully'
        };
      }
      return { success: false, message: 'Contact not found' };
    },

    async delete(id: string): Promise<ApiResponse<void>> {
      await delay(300);
      const index = mockContacts.findIndex(c => c.id === id);
      if (index !== -1) {
        mockContacts.splice(index, 1);
        return { success: true, message: 'Contact deleted successfully' };
      }
      return { success: false, message: 'Contact not found' };
    }
  },

  accounts: {
    async getAll(): Promise<PaginatedResponse<Account>> {
      await delay(300);
      return {
        success: true,
        data: mockAccounts,
        pagination: {
          page: 1,
          limit: 10,
          total: mockAccounts.length,
          totalPages: 1
        }
      };
    },

    async getById(id: string): Promise<ApiResponse<Account>> {
      await delay(200);
      const account = mockAccounts.find(a => a.id === id);
      return {
        success: !!account,
        data: account,
        message: account ? 'Account found' : 'Account not found'
      };
    },

    async create(data: any): Promise<ApiResponse<Account>> {
      await delay(400);
      const newAccount: Account = {
        id: String(mockAccounts.length + 1),
        ...data,
        createdAt: new Date(),
        updatedAt: new Date(),
        createdBy: 'demo-user',
        updatedBy: 'demo-user'
      };
      mockAccounts.push(newAccount);
      return {
        success: true,
        data: newAccount,
        message: 'Account created successfully'
      };
    },

    async update(id: string, data: any): Promise<ApiResponse<Account>> {
      await delay(400);
      const index = mockAccounts.findIndex(a => a.id === id);
      if (index !== -1) {
        mockAccounts[index] = { ...mockAccounts[index], ...data, updatedAt: new Date() };
        return {
          success: true,
          data: mockAccounts[index],
          message: 'Account updated successfully'
        };
      }
      return { success: false, message: 'Account not found' };
    },

    async delete(id: string): Promise<ApiResponse<void>> {
      await delay(300);
      const index = mockAccounts.findIndex(a => a.id === id);
      if (index !== -1) {
        mockAccounts.splice(index, 1);
        return { success: true, message: 'Account deleted successfully' };
      }
      return { success: false, message: 'Account not found' };
    }
  },

  deals: {
    async getAll(): Promise<PaginatedResponse<ActiveDeal>> {
      await delay(300);
      return {
        success: true,
        data: mockDeals,
        pagination: {
          page: 1,
          limit: 10,
          total: mockDeals.length,
          totalPages: 1
        }
      };
    },

    async getById(id: string): Promise<ApiResponse<ActiveDeal>> {
      await delay(200);
      const deal = mockDeals.find(d => d.id === id);
      return {
        success: !!deal,
        data: deal,
        message: deal ? 'Deal found' : 'Deal not found'
      };
    },

    async create(data: any): Promise<ApiResponse<ActiveDeal>> {
      await delay(400);
      const newDeal: ActiveDeal = {
        id: String(mockDeals.length + 1),
        ...data,
        createdAt: new Date(),
        updatedAt: new Date(),
        createdBy: 'demo-user',
        updatedBy: 'demo-user'
      };
      mockDeals.push(newDeal);
      return {
        success: true,
        data: newDeal,
        message: 'Deal created successfully'
      };
    },

    async update(id: string, data: any): Promise<ApiResponse<ActiveDeal>> {
      await delay(400);
      const index = mockDeals.findIndex(d => d.id === id);
      if (index !== -1) {
        mockDeals[index] = { ...mockDeals[index], ...data, updatedAt: new Date() };
        return {
          success: true,
          data: mockDeals[index],
          message: 'Deal updated successfully'
        };
      }
      return { success: false, message: 'Deal not found' };
    },

    async delete(id: string): Promise<ApiResponse<void>> {
      await delay(300);
      const index = mockDeals.findIndex(d => d.id === id);
      if (index !== -1) {
        mockDeals.splice(index, 1);
        return { success: true, message: 'Deal deleted successfully' };
      }
      return { success: false, message: 'Deal not found' };
    }
  },

  leads: {
    async getAll(): Promise<PaginatedResponse<Lead>> {
      await delay(300);
      return {
        success: true,
        data: mockLeads,
        pagination: {
          page: 1,
          limit: 10,
          total: mockLeads.length,
          totalPages: 1
        }
      };
    },

    async getById(id: string): Promise<ApiResponse<Lead>> {
      await delay(200);
      const lead = mockLeads.find(l => l.id === id);
      return {
        success: !!lead,
        data: lead,
        message: lead ? 'Lead found' : 'Lead not found'
      };
    },

    async create(data: any): Promise<ApiResponse<Lead>> {
      await delay(400);
      const newLead: Lead = {
        id: String(mockLeads.length + 1),
        ...data,
        createdAt: new Date(),
        updatedAt: new Date(),
        createdBy: 'demo-user',
        updatedBy: 'demo-user'
      };
      mockLeads.push(newLead);
      return {
        success: true,
        data: newLead,
        message: 'Lead created successfully'
      };
    },

    async update(id: string, data: any): Promise<ApiResponse<Lead>> {
      await delay(400);
      const index = mockLeads.findIndex(l => l.id === id);
      if (index !== -1) {
        mockLeads[index] = { ...mockLeads[index], ...data, updatedAt: new Date() };
        return {
          success: true,
          data: mockLeads[index],
          message: 'Lead updated successfully'
        };
      }
      return { success: false, message: 'Lead not found' };
    },

    async delete(id: string): Promise<ApiResponse<void>> {
      await delay(300);
      const index = mockLeads.findIndex(l => l.id === id);
      if (index !== -1) {
        mockLeads.splice(index, 1);
        return { success: true, message: 'Lead deleted successfully' };
      }
      return { success: false, message: 'Lead not found' };
    }
  },

  activities: {
    async getAll(): Promise<PaginatedResponse<ActivityLog>> {
      await delay(300);
      return {
        success: true,
        data: mockActivities,
        pagination: {
          page: 1,
          limit: 10,
          total: mockActivities.length,
          totalPages: 1
        }
      };
    },

    async getById(id: string): Promise<ApiResponse<ActivityLog>> {
      await delay(200);
      const activity = mockActivities.find(a => a.id === id);
      return {
        success: !!activity,
        data: activity,
        message: activity ? 'Activity found' : 'Activity not found'
      };
    },

    async create(data: any): Promise<ApiResponse<ActivityLog>> {
      await delay(400);
      const newActivity: ActivityLog = {
        id: String(mockActivities.length + 1),
        ...data,
        createdAt: new Date(),
        updatedAt: new Date(),
        createdBy: 'demo-user',
        updatedBy: 'demo-user'
      };
      mockActivities.push(newActivity);
      return {
        success: true,
        data: newActivity,
        message: 'Activity created successfully'
      };
    },

    async update(id: string, data: any): Promise<ApiResponse<ActivityLog>> {
      await delay(400);
      const index = mockActivities.findIndex(a => a.id === id);
      if (index !== -1) {
        mockActivities[index] = { ...mockActivities[index], ...data, updatedAt: new Date() };
        return {
          success: true,
          data: mockActivities[index],
          message: 'Activity updated successfully'
        };
      }
      return { success: false, message: 'Activity not found' };
    },

    async delete(id: string): Promise<ApiResponse<void>> {
      await delay(300);
      const index = mockActivities.findIndex(a => a.id === id);
      if (index !== -1) {
        mockActivities.splice(index, 1);
        return { success: true, message: 'Activity deleted successfully' };
      }
      return { success: false, message: 'Activity not found' };
    }
  },

  profile: {
    async getCurrent(): Promise<ApiResponse<UserProfile>> {
      await delay(200);
      return {
        success: true,
        data: mockUserProfile,
        message: 'Profile loaded successfully'
      };
    },

    async updateCurrent(data: any): Promise<ApiResponse<UserProfile>> {
      await delay(400);
      Object.assign(mockUserProfile, data, { updatedAt: new Date() });
      return {
        success: true,
        data: mockUserProfile,
        message: 'Profile updated successfully'
      };
    },

    async getById(id: string): Promise<ApiResponse<UserProfile>> {
      await delay(200);
      return {
        success: true,
        data: mockUserProfile,
        message: 'Profile loaded successfully'
      };
    },

    async update(id: string, data: any): Promise<ApiResponse<UserProfile>> {
      await delay(400);
      Object.assign(mockUserProfile, data, { updatedAt: new Date() });
      return {
        success: true,
        data: mockUserProfile,
        message: 'Profile updated successfully'
      };
    }
  },

  reports: {
    async generate(data: any): Promise<ApiResponse<any>> {
      await delay(1000);
      return {
        success: true,
        data: {
          reportId: 'demo-report-' + Date.now(),
          reportType: data.reportType,
          period: data.period,
          metrics: {
            totalContacts: mockContacts.length,
            totalAccounts: mockAccounts.length,
            totalDeals: mockDeals.length,
            totalRevenue: mockDeals
              .filter(d => d.stage === 'Order Won')
              .reduce((sum, d) => sum + parseFloat(d.dealValue?.replace(/[^0-9.-]+/g, '') || '0'), 0)
          },
          generatedAt: new Date()
        },
        message: 'Report generated successfully'
      };
    },

    async download(id: string): Promise<ApiResponse<any>> {
      await delay(500);
      return {
        success: true,
        data: { downloadUrl: '#', message: 'Demo report download' },
        message: 'Report download ready'
      };
    }
  }
};
