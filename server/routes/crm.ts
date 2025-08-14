import { RequestHandler } from "express";
import { 
  Contact, Account, ActivityLog, ActiveDeal, Lead, ReportData,
  ApiResponse, PaginatedResponse,
  CreateContactRequest, UpdateContactRequest,
  CreateAccountRequest, UpdateAccountRequest,
  CreateActivityRequest, UpdateActivityRequest,
  CreateDealRequest, UpdateDealRequest,
  CreateLeadRequest, UpdateLeadRequest,
  ReportRequest
} from "@shared/models";

// Mock database - In production, replace with actual database connection
let contacts: Contact[] = [];
let accounts: Account[] = [];
let activities: ActivityLog[] = [];
let deals: ActiveDeal[] = [];
let leads: Lead[] = [];

// Utility functions
const generateId = () => Math.random().toString(36).substr(2, 9);
const getCurrentUser = () => "current-user"; // In production, get from authentication

// CONTACTS API
export const getContacts: RequestHandler = (req, res) => {
  const page = parseInt(req.query.page as string) || 1;
  const limit = parseInt(req.query.limit as string) || 50;
  const search = req.query.search as string;
  
  let filteredContacts = contacts;
  
  if (search) {
    filteredContacts = contacts.filter(contact => 
      Object.values(contact).some(value => 
        String(value).toLowerCase().includes(search.toLowerCase())
      )
    );
  }
  
  const total = filteredContacts.length;
  const totalPages = Math.ceil(total / limit);
  const startIndex = (page - 1) * limit;
  const endIndex = startIndex + limit;
  
  const paginatedContacts = filteredContacts.slice(startIndex, endIndex);
  
  const response: PaginatedResponse<Contact> = {
    success: true,
    data: paginatedContacts,
    pagination: { page, limit, total, totalPages }
  };
  
  res.json(response);
};

export const getContact: RequestHandler = (req, res) => {
  const contact = contacts.find(c => c.id === req.params.id);
  
  if (!contact) {
    return res.status(404).json({ success: false, message: "Contact not found" });
  }
  
  res.json({ success: true, data: contact });
};

export const createContact: RequestHandler = (req, res) => {
  const contactData: CreateContactRequest = req.body;
  const currentUser = getCurrentUser();
  const now = new Date();
  
  const newContact: Contact = {
    ...contactData,
    id: generateId(),
    createdAt: now,
    updatedAt: now,
    createdBy: currentUser,
    updatedBy: currentUser
  };
  
  contacts.push(newContact);
  
  res.status(201).json({ success: true, data: newContact });
};

export const updateContact: RequestHandler = (req, res) => {
  const contactIndex = contacts.findIndex(c => c.id === req.params.id);
  
  if (contactIndex === -1) {
    return res.status(404).json({ success: false, message: "Contact not found" });
  }
  
  const updateData: UpdateContactRequest = req.body;
  const currentUser = getCurrentUser();
  
  contacts[contactIndex] = {
    ...contacts[contactIndex],
    ...updateData,
    updatedAt: new Date(),
    updatedBy: currentUser
  };
  
  res.json({ success: true, data: contacts[contactIndex] });
};

export const deleteContact: RequestHandler = (req, res) => {
  const requestedId = req.params.id;
  console.log(`Attempting to delete contact with ID: ${requestedId}`);
  console.log(`Available contact IDs: ${contacts.map(c => c.id).join(', ')}`);

  const contactIndex = contacts.findIndex(c => c.id === requestedId);

  if (contactIndex === -1) {
    console.log(`Contact not found with ID: ${requestedId}`);
    return res.status(404).json({ success: false, message: "Contact not found" });
  }

  console.log(`Deleting contact: ${contacts[contactIndex].firstName} ${contacts[contactIndex].lastName}`);
  contacts.splice(contactIndex, 1);
  res.json({ success: true, message: "Contact deleted successfully" });
};

// ACCOUNTS API
export const getAccounts: RequestHandler = (req, res) => {
  console.log('getAccounts called');
  console.log('Total accounts in memory:', accounts.length);
  console.log('Account IDs:', accounts.map(a => a.id));

  const page = parseInt(req.query.page as string) || 1;
  const limit = parseInt(req.query.limit as string) || 50;
  const search = req.query.search as string;

  let filteredAccounts = accounts;

  if (search) {
    filteredAccounts = accounts.filter(account =>
      Object.values(account).some(value =>
        String(value).toLowerCase().includes(search.toLowerCase())
      )
    );
  }

  const total = filteredAccounts.length;
  const totalPages = Math.ceil(total / limit);
  const startIndex = (page - 1) * limit;
  const endIndex = startIndex + limit;

  const paginatedAccounts = filteredAccounts.slice(startIndex, endIndex);

  const response: PaginatedResponse<Account> = {
    success: true,
    data: paginatedAccounts,
    pagination: { page, limit, total, totalPages }
  };

  console.log('Sending response with', paginatedAccounts.length, 'accounts');
  res.json(response);
};

export const getAccount: RequestHandler = (req, res) => {
  const account = accounts.find(a => a.id === req.params.id);
  
  if (!account) {
    return res.status(404).json({ success: false, message: "Account not found" });
  }
  
  res.json({ success: true, data: account });
};

export const createAccount: RequestHandler = (req, res) => {
  const accountData: CreateAccountRequest = req.body;
  const currentUser = getCurrentUser();
  const now = new Date();
  
  const newAccount: Account = {
    ...accountData,
    id: generateId(),
    createdAt: now,
    updatedAt: now,
    createdBy: currentUser,
    updatedBy: currentUser
  };
  
  accounts.push(newAccount);
  
  res.status(201).json({ success: true, data: newAccount });
};

export const updateAccount: RequestHandler = (req, res) => {
  const accountIndex = accounts.findIndex(a => a.id === req.params.id);
  
  if (accountIndex === -1) {
    return res.status(404).json({ success: false, message: "Account not found" });
  }
  
  const updateData: UpdateAccountRequest = req.body;
  const currentUser = getCurrentUser();
  
  accounts[accountIndex] = {
    ...accounts[accountIndex],
    ...updateData,
    updatedAt: new Date(),
    updatedBy: currentUser
  };
  
  res.json({ success: true, data: accounts[accountIndex] });
};

export const deleteAccount: RequestHandler = (req, res) => {
  const requestedId = req.params.id;
  console.log(`Attempting to delete account with ID: ${requestedId}`);
  console.log(`Available account IDs: ${accounts.map(a => a.id).join(', ')}`);

  const accountIndex = accounts.findIndex(a => a.id === requestedId);

  if (accountIndex === -1) {
    console.log(`Account not found with ID: ${requestedId}`);
    return res.status(404).json({ success: false, message: "Account not found" });
  }

  console.log(`Deleting account: ${accounts[accountIndex].accountName}`);
  accounts.splice(accountIndex, 1);
  res.json({ success: true, message: "Account deleted successfully" });
};

// ACTIVITIES API
export const getActivities: RequestHandler = (req, res) => {
  const page = parseInt(req.query.page as string) || 1;
  const limit = parseInt(req.query.limit as string) || 50;

  const total = activities.length;
  const totalPages = Math.ceil(total / limit);
  const startIndex = (page - 1) * limit;
  const endIndex = startIndex + limit;

  const paginatedActivities = activities.slice(startIndex, endIndex);

  const response: PaginatedResponse<ActivityLog> = {
    success: true,
    data: paginatedActivities,
    pagination: { page, limit, total, totalPages }
  };

  res.json(response);
};

export const getActivity: RequestHandler = (req, res) => {
  const activity = activities.find(a => a.id === req.params.id);

  if (!activity) {
    return res.status(404).json({ success: false, message: "Activity not found" });
  }

  res.json({ success: true, data: activity });
};

export const createActivity: RequestHandler = (req, res) => {
  const activityData: CreateActivityRequest = req.body;
  const currentUser = getCurrentUser();
  const now = new Date();

  const newActivity: ActivityLog = {
    ...activityData,
    id: generateId(),
    createdAt: now,
    updatedAt: now,
    createdBy: currentUser,
    updatedBy: currentUser
  };

  activities.push(newActivity);

  res.status(201).json({ success: true, data: newActivity });
};

export const updateActivity: RequestHandler = (req, res) => {
  const activityIndex = activities.findIndex(a => a.id === req.params.id);

  if (activityIndex === -1) {
    return res.status(404).json({ success: false, message: "Activity not found" });
  }

  const updateData: UpdateActivityRequest = req.body;
  const currentUser = getCurrentUser();

  activities[activityIndex] = {
    ...activities[activityIndex],
    ...updateData,
    updatedAt: new Date(),
    updatedBy: currentUser
  };

  res.json({ success: true, data: activities[activityIndex] });
};

export const deleteActivity: RequestHandler = (req, res) => {
  const activityIndex = activities.findIndex(a => a.id === req.params.id);

  if (activityIndex === -1) {
    return res.status(404).json({ success: false, message: "Activity not found" });
  }

  activities.splice(activityIndex, 1);
  res.json({ success: true, message: "Activity deleted successfully" });
};

// DEALS API
export const getDeals: RequestHandler = (req, res) => {
  const page = parseInt(req.query.page as string) || 1;
  const limit = parseInt(req.query.limit as string) || 50;

  const total = deals.length;
  const totalPages = Math.ceil(total / limit);
  const startIndex = (page - 1) * limit;
  const endIndex = startIndex + limit;

  const paginatedDeals = deals.slice(startIndex, endIndex);

  const response: PaginatedResponse<ActiveDeal> = {
    success: true,
    data: paginatedDeals,
    pagination: { page, limit, total, totalPages }
  };

  res.json(response);
};

export const getDeal: RequestHandler = (req, res) => {
  const deal = deals.find(d => d.id === req.params.id);

  if (!deal) {
    return res.status(404).json({ success: false, message: "Deal not found" });
  }

  res.json({ success: true, data: deal });
};

export const createDeal: RequestHandler = (req, res) => {
  const dealData: CreateDealRequest = req.body;
  const currentUser = getCurrentUser();
  const now = new Date();

  const newDeal: ActiveDeal = {
    ...dealData,
    id: generateId(),
    createdAt: now,
    updatedAt: now,
    createdBy: currentUser,
    updatedBy: currentUser
  };

  deals.push(newDeal);

  res.status(201).json({ success: true, data: newDeal });
};

export const updateDeal: RequestHandler = (req, res) => {
  const dealIndex = deals.findIndex(d => d.id === req.params.id);

  if (dealIndex === -1) {
    return res.status(404).json({ success: false, message: "Deal not found" });
  }

  const updateData: UpdateDealRequest = req.body;
  const currentUser = getCurrentUser();

  deals[dealIndex] = {
    ...deals[dealIndex],
    ...updateData,
    updatedAt: new Date(),
    updatedBy: currentUser
  };

  res.json({ success: true, data: deals[dealIndex] });
};

export const deleteDeal: RequestHandler = (req, res) => {
  const dealIndex = deals.findIndex(d => d.id === req.params.id);

  if (dealIndex === -1) {
    return res.status(404).json({ success: false, message: "Deal not found" });
  }

  deals.splice(dealIndex, 1);
  res.json({ success: true, message: "Deal deleted successfully" });
};

// LEADS API
export const getLeads: RequestHandler = (req, res) => {
  const page = parseInt(req.query.page as string) || 1;
  const limit = parseInt(req.query.limit as string) || 50;

  const total = leads.length;
  const totalPages = Math.ceil(total / limit);
  const startIndex = (page - 1) * limit;
  const endIndex = startIndex + limit;

  const paginatedLeads = leads.slice(startIndex, endIndex);

  const response: PaginatedResponse<Lead> = {
    success: true,
    data: paginatedLeads,
    pagination: { page, limit, total, totalPages }
  };

  res.json(response);
};

export const getLead: RequestHandler = (req, res) => {
  const lead = leads.find(l => l.id === req.params.id);

  if (!lead) {
    return res.status(404).json({ success: false, message: "Lead not found" });
  }

  res.json({ success: true, data: lead });
};

export const createLead: RequestHandler = (req, res) => {
  const leadData: CreateLeadRequest = req.body;
  const currentUser = getCurrentUser();
  const now = new Date();

  const newLead: Lead = {
    ...leadData,
    id: generateId(),
    createdAt: now,
    updatedAt: now,
    createdBy: currentUser,
    updatedBy: currentUser
  };

  leads.push(newLead);

  res.status(201).json({ success: true, data: newLead });
};

export const updateLead: RequestHandler = (req, res) => {
  const leadIndex = leads.findIndex(l => l.id === req.params.id);

  if (leadIndex === -1) {
    return res.status(404).json({ success: false, message: "Lead not found" });
  }

  const updateData: UpdateLeadRequest = req.body;
  const currentUser = getCurrentUser();

  leads[leadIndex] = {
    ...leads[leadIndex],
    ...updateData,
    updatedAt: new Date(),
    updatedBy: currentUser
  };

  res.json({ success: true, data: leads[leadIndex] });
};

export const deleteLead: RequestHandler = (req, res) => {
  const leadIndex = leads.findIndex(l => l.id === req.params.id);

  if (leadIndex === -1) {
    return res.status(404).json({ success: false, message: "Lead not found" });
  }

  leads.splice(leadIndex, 1);
  res.json({ success: true, message: "Lead deleted successfully" });
};

// REPORTS API
export const generateReport: RequestHandler = (req, res) => {
  const reportRequest: ReportRequest = req.body;
  const currentUser = getCurrentUser();
  
  // Generate mock report data based on request
  const reportData: ReportData = {
    id: generateId(),
    reportType: reportRequest.reportType,
    period: reportRequest.period,
    salesRep: reportRequest.salesRep,
    geo: reportRequest.geo as any,
    businessLine: reportRequest.businessLine,
    org: reportRequest.org,
    metrics: {
      meetingsFixed: Math.floor(Math.random() * 50),
      meetingsCompleted: Math.floor(Math.random() * 40),
      opportunityCreatedNos: Math.floor(Math.random() * 20),
      opportunityCreatedValue: Math.floor(Math.random() * 100000),
      proposalSubmittedNos: Math.floor(Math.random() * 15),
      proposalSubmittedValue: Math.floor(Math.random() * 75000),
      orderWonNos: Math.floor(Math.random() * 10),
      orderWonValue: Math.floor(Math.random() * 50000),
      orderLostNos: Math.floor(Math.random() * 5),
      orderLostValue: Math.floor(Math.random() * 25000),
      meetingsFixedVsTarget: Math.floor(Math.random() * 120),
      meetingsCompletedVsTarget: Math.floor(Math.random() * 110),
      opportunityCreatedVsTarget: Math.floor(Math.random() * 105),
      opportunityCreatedValueVsTarget: Math.floor(Math.random() * 115),
      proposalSubmittedVsTarget: Math.floor(Math.random() * 95),
      proposalSubmittedValueVsTarget: Math.floor(Math.random() * 100),
      orderWonVsTarget: Math.floor(Math.random() * 130),
      orderWonValueVsTarget: Math.floor(Math.random() * 125),
    },
    generatedAt: new Date(),
    generatedBy: currentUser
  };
  
  if (reportRequest.format === 'excel') {
    // In production, generate actual Excel file
    res.setHeader('Content-Disposition', 'attachment; filename=report.xlsx');
    res.setHeader('Content-Type', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet');
    res.json({ success: true, data: reportData, downloadUrl: '/api/reports/download/' + reportData.id });
  } else if (reportRequest.format === 'pdf') {
    // In production, generate actual PDF file
    res.setHeader('Content-Disposition', 'attachment; filename=report.pdf');
    res.setHeader('Content-Type', 'application/pdf');
    res.json({ success: true, data: reportData, downloadUrl: '/api/reports/download/' + reportData.id });
  } else {
    res.json({ success: true, data: reportData });
  }
};

export const downloadReport: RequestHandler = (req, res) => {
  const reportId = req.params.id;
  // In production, retrieve and send the actual file
  res.json({ success: true, message: "Report download would start here", reportId });
};

// Initialize with sample data
export const initializeSampleData = () => {
  // Sample Contacts
  contacts = [
    {
      id: "contact1",
      firstName: "John",
      lastName: "Smith",
      title: "CEO",
      associatedAccount: "Acme Corp",
      emailAddress: "john@acme.com",
      deskPhone: "(555) 123-4567",
      mobilePhone: "(555) 123-4568",
      city: "New York",
      state: "NY",
      country: "USA",
      timeZone: "EST",
      source: "Referral",
      owner: "Sales Rep 1",
      status: "Prospect",
      createdAt: new Date(),
      updatedAt: new Date(),
      createdBy: "system",
      updatedBy: "system"
    },
    {
      id: "contact2",
      firstName: "Sarah",
      lastName: "Johnson",
      title: "CTO",
      associatedAccount: "Tech Solutions",
      emailAddress: "sarah@techsol.com",
      deskPhone: "(555) 987-6543",
      mobilePhone: "(555) 987-6544",
      city: "San Francisco",
      state: "CA",
      country: "USA",
      timeZone: "PST",
      source: "Data Research",
      owner: "Sales Rep 2",
      status: "Active Deal",
      createdAt: new Date(),
      updatedAt: new Date(),
      createdBy: "system",
      updatedBy: "system"
    }
  ];

  // Sample Accounts
  accounts = [
    {
      id: "account1",
      accountName: "Acme Corporation",
      accountRating: "Platinum (Must Have)",
      accountOwner: "Sales Rep 1",
      status: "Prospect",
      industry: "Technology",
      revenue: "$10M",
      numberOfEmployees: "500",
      addressLine1: "123 Business Ave",
      city: "New York",
      state: "NY",
      country: "USA",
      zipPostCode: "10001",
      timeZone: "EST",
      website: "acme.com",
      geo: "Americas",
      createdAt: new Date(),
      updatedAt: new Date(),
      createdBy: "system",
      updatedBy: "system"
    },
    {
      id: "account2",
      accountName: "Tech Solutions Inc",
      accountRating: "Gold (High Priority)",
      accountOwner: "Sales Rep 2",
      status: "Active Deal",
      industry: "Software",
      revenue: "$5M",
      numberOfEmployees: "250",
      addressLine1: "456 Tech Street",
      city: "San Francisco",
      state: "CA",
      country: "USA",
      zipPostCode: "94105",
      timeZone: "PST",
      website: "techsol.com",
      geo: "Americas",
      createdAt: new Date(),
      updatedAt: new Date(),
      createdBy: "system",
      updatedBy: "system"
    }
  ];

  // Sample Leads
  leads = [
    {
      id: "lead1",
      firstName: "Mike",
      lastName: "Wilson",
      company: "StartupCo",
      title: "Founder",
      phone: "(555) 111-2222",
      email: "mike@startup.co",
      leadSource: "Website",
      status: "New",
      rating: "Hot",
      owner: "Sales Rep 1",
      createdAt: new Date(),
      updatedAt: new Date(),
      createdBy: "system",
      updatedBy: "system"
    }
  ];
};
