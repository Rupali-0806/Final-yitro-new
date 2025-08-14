import { RequestHandler } from "express";
import { prisma } from "../lib/prisma";
import {
  Contact,
  Account,
  ActivityLog,
  ActiveDeal,
  Lead,
  UserProfile,
  ApiResponse,
  PaginatedResponse,
  CreateContactRequest,
  UpdateContactRequest,
  CreateAccountRequest,
  UpdateAccountRequest,
  CreateActivityRequest,
  UpdateActivityRequest,
  CreateDealRequest,
  UpdateDealRequest,
  CreateLeadRequest,
  UpdateLeadRequest,
  CreateUserProfileRequest,
  UpdateUserProfileRequest,
  ReportRequest,
} from "@shared/models";

// Helper function to map enum values
const mapEnumValue = (value: any, enumMapping: Record<string, string>) => {
  if (!value) return undefined;
  return (
    Object.keys(enumMapping).find((key) => enumMapping[key] === value) || value
  );
};

// Contacts Routes
export const getContacts: RequestHandler = async (req, res) => {
  try {
    const page = parseInt(req.query.page as string) || 1;
    const limit = parseInt(req.query.limit as string) || 10;
    const search = req.query.search as string;

    const skip = (page - 1) * limit;

    const where = search
      ? {
          OR: [
            { firstName: { contains: search, mode: "insensitive" as const } },
            { lastName: { contains: search, mode: "insensitive" as const } },
            {
              emailAddress: { contains: search, mode: "insensitive" as const },
            },
          ],
        }
      : {};

    const [contacts, total] = await Promise.all([
      prisma.contact.findMany({
        where,
        skip,
        take: limit,
        orderBy: { createdAt: "desc" },
      }),
      prisma.contact.count({ where }),
    ]);

    const response: PaginatedResponse<Contact> = {
      success: true,
      data: contacts.map((contact) => ({
        ...contact,
        source: contact.source?.replace(/_/g, " "),
        status: contact.status?.replace(/_/g, " "),
      })) as Contact[],
      pagination: {
        page,
        limit,
        total,
        totalPages: Math.ceil(total / limit),
      },
    };

    res.json(response);
  } catch (error) {
    console.error("Error fetching contacts:", error);
    res.status(500).json({ success: false, error: "Failed to fetch contacts" });
  }
};

export const getContact: RequestHandler = async (req, res) => {
  try {
    const contact = await prisma.contact.findUnique({
      where: { id: req.params.id },
      include: {
        account: true,
        activities: true,
        deals: true,
      },
    });

    if (!contact) {
      return res
        .status(404)
        .json({ success: false, message: "Contact not found" });
    }

    const response: ApiResponse<Contact> = {
      success: true,
      data: {
        ...contact,
        source: contact.source?.replace(/_/g, " "),
        status: contact.status?.replace(/_/g, " "),
      } as Contact,
    };

    res.json(response);
  } catch (error) {
    console.error("Error fetching contact:", error);
    res.status(500).json({ success: false, error: "Failed to fetch contact" });
  }
};

export const createContact: RequestHandler = async (req, res) => {
  try {
    const data: CreateContactRequest = req.body;

    const contact = await prisma.contact.create({
      data: {
        ...data,
        source: data.source
          ? data.source.replace(/\s+/g, "_").toUpperCase()
          : undefined,
        status: data.status
          ? data.status.replace(/\s+/g, "_").toUpperCase()
          : undefined,
        createdBy: "system",
        updatedBy: "system",
      } as any,
    });

    const response: ApiResponse<Contact> = {
      success: true,
      data: {
        ...contact,
        source: contact.source?.replace(/_/g, " "),
        status: contact.status?.replace(/_/g, " "),
      } as Contact,
      message: "Contact created successfully",
    };

    res.status(201).json(response);
  } catch (error) {
    console.error("Error creating contact:", error);
    res.status(500).json({ success: false, error: "Failed to create contact" });
  }
};

export const updateContact: RequestHandler = async (req, res) => {
  try {
    const data: UpdateContactRequest = req.body;

    const contact = await prisma.contact.update({
      where: { id: req.params.id },
      data: {
        ...data,
        source: data.source
          ? data.source.replace(/\s+/g, "_").toUpperCase()
          : undefined,
        status: data.status
          ? data.status.replace(/\s+/g, "_").toUpperCase()
          : undefined,
        updatedBy: "system",
      } as any,
    });

    const response: ApiResponse<Contact> = {
      success: true,
      data: {
        ...contact,
        source: contact.source?.replace(/_/g, " "),
        status: contact.status?.replace(/_/g, " "),
      } as Contact,
      message: "Contact updated successfully",
    };

    res.json(response);
  } catch (error) {
    console.error("Error updating contact:", error);
    res.status(500).json({ success: false, error: "Failed to update contact" });
  }
};

export const deleteContact: RequestHandler = async (req, res) => {
  try {
    await prisma.contact.delete({
      where: { id: req.params.id },
    });

    res.json({ success: true, message: "Contact deleted successfully" });
  } catch (error) {
    console.error("Error deleting contact:", error);
    res.status(500).json({ success: false, error: "Failed to delete contact" });
  }
};

// Accounts Routes
export const getAccounts: RequestHandler = async (req, res) => {
  try {
    const page = parseInt(req.query.page as string) || 1;
    const limit = parseInt(req.query.limit as string) || 10;
    const search = req.query.search as string;

    const skip = (page - 1) * limit;

    const where = search
      ? {
          OR: [
            { accountName: { contains: search, mode: "insensitive" as const } },
            { industry: { contains: search, mode: "insensitive" as const } },
          ],
        }
      : {};

    const [accounts, total] = await Promise.all([
      prisma.account.findMany({
        where,
        skip,
        take: limit,
        orderBy: { createdAt: "desc" },
      }),
      prisma.account.count({ where }),
    ]);

    const response: PaginatedResponse<Account> = {
      success: true,
      data: accounts.map((account) => ({
        ...account,
        accountRating: account.accountRating
          ? account.accountRating.replace(/_/g, " ")
          : undefined,
        status: account.status ? account.status.replace("_", " ") : undefined,
        geo: account.geo ? account.geo.replace("_", " ") : undefined,
      })) as Account[],
      pagination: {
        page,
        limit,
        total,
        totalPages: Math.ceil(total / limit),
      },
    };

    res.json(response);
  } catch (error) {
    console.error("Error fetching accounts:", error);
    res.status(500).json({ success: false, error: "Failed to fetch accounts" });
  }
};

export const getAccount: RequestHandler = async (req, res) => {
  try {
    const account = await prisma.account.findUnique({
      where: { id: req.params.id },
      include: {
        contacts: true,
        activities: true,
        deals: true,
      },
    });

    if (!account) {
      return res
        .status(404)
        .json({ success: false, message: "Account not found" });
    }

    const response: ApiResponse<Account> = {
      success: true,
      data: {
        ...account,
        accountRating: account.accountRating
          ? account.accountRating.replace(/_/g, " ")
          : undefined,
        status: account.status ? account.status.replace("_", " ") : undefined,
        geo: account.geo ? account.geo.replace("_", " ") : undefined,
      } as Account,
    };

    res.json(response);
  } catch (error) {
    console.error("Error fetching account:", error);
    res.status(500).json({ success: false, error: "Failed to fetch account" });
  }
};

export const createAccount: RequestHandler = async (req, res) => {
  try {
    const data: CreateAccountRequest = req.body;

    const account = await prisma.account.create({
      data: {
        ...data,
        accountRating: data.accountRating
          ? data.accountRating.replace(/\s+/g, "_").toUpperCase()
          : undefined,
        status: data.status
          ? data.status.replace(" ", "_").toUpperCase()
          : undefined,
        geo: data.geo ? data.geo.replace(" ", "_").toUpperCase() : undefined,
        createdBy: "system",
        updatedBy: "system",
      } as any,
    });

    const response: ApiResponse<Account> = {
      success: true,
      data: {
        ...account,
        accountRating: account.accountRating
          ? account.accountRating.replace(/_/g, " ")
          : undefined,
        status: account.status ? account.status.replace("_", " ") : undefined,
        geo: account.geo ? account.geo.replace("_", " ") : undefined,
      } as Account,
      message: "Account created successfully",
    };

    res.status(201).json(response);
  } catch (error) {
    console.error("Error creating account:", error);
    res.status(500).json({ success: false, error: "Failed to create account" });
  }
};

export const updateAccount: RequestHandler = async (req, res) => {
  try {
    const data: UpdateAccountRequest = req.body;

    const account = await prisma.account.update({
      where: { id: req.params.id },
      data: {
        ...data,
        accountRating: data.accountRating
          ? data.accountRating.replace(/\s+/g, "_").toUpperCase()
          : undefined,
        status: data.status
          ? data.status.replace(" ", "_").toUpperCase()
          : undefined,
        geo: data.geo ? data.geo.replace(" ", "_").toUpperCase() : undefined,
        updatedBy: "system",
      } as any,
    });

    const response: ApiResponse<Account> = {
      success: true,
      data: {
        ...account,
        accountRating: account.accountRating
          ? account.accountRating.replace(/_/g, " ")
          : undefined,
        status: account.status ? account.status.replace("_", " ") : undefined,
        geo: account.geo ? account.geo.replace("_", " ") : undefined,
      } as Account,
      message: "Account updated successfully",
    };

    res.json(response);
  } catch (error) {
    console.error("Error updating account:", error);
    res.status(500).json({ success: false, error: "Failed to update account" });
  }
};

export const deleteAccount: RequestHandler = async (req, res) => {
  try {
    await prisma.account.delete({
      where: { id: req.params.id },
    });

    res.json({ success: true, message: "Account deleted successfully" });
  } catch (error) {
    console.error("Error deleting account:", error);
    res.status(500).json({ success: false, error: "Failed to delete account" });
  }
};

// Activities Routes
export const getActivities: RequestHandler = async (req, res) => {
  try {
    const page = parseInt(req.query.page as string) || 1;
    const limit = parseInt(req.query.limit as string) || 10;

    const skip = (page - 1) * limit;

    const [activities, total] = await Promise.all([
      prisma.activityLog.findMany({
        skip,
        take: limit,
        orderBy: { dateTime: "desc" },
        include: {
          contact: true,
          account: true,
        },
      }),
      prisma.activityLog.count(),
    ]);

    const response: PaginatedResponse<ActivityLog> = {
      success: true,
      data: activities.map((activity) => ({
        ...activity,
        activityType: activity.activityType.replace("_", " "),
        outcomeDisposition: activity.outcomeDisposition
          ? activity.outcomeDisposition.replace(/_/g, " ")
          : undefined,
      })) as ActivityLog[],
      pagination: {
        page,
        limit,
        total,
        totalPages: Math.ceil(total / limit),
      },
    };

    res.json(response);
  } catch (error) {
    console.error("Error fetching activities:", error);
    res
      .status(500)
      .json({ success: false, error: "Failed to fetch activities" });
  }
};

export const getActivity: RequestHandler = async (req, res) => {
  try {
    const activity = await prisma.activityLog.findUnique({
      where: { id: req.params.id },
      include: {
        contact: true,
        account: true,
      },
    });

    if (!activity) {
      return res
        .status(404)
        .json({ success: false, message: "Activity not found" });
    }

    const response: ApiResponse<ActivityLog> = {
      success: true,
      data: {
        ...activity,
        activityType: activity.activityType.replace("_", " "),
        outcomeDisposition: activity.outcomeDisposition
          ? activity.outcomeDisposition.replace(/_/g, " ")
          : undefined,
      } as ActivityLog,
    };

    res.json(response);
  } catch (error) {
    console.error("Error fetching activity:", error);
    res.status(500).json({ success: false, error: "Failed to fetch activity" });
  }
};

export const createActivity: RequestHandler = async (req, res) => {
  try {
    const data: CreateActivityRequest = req.body;

    const activity = await prisma.activityLog.create({
      data: {
        ...data,
        activityType: data.activityType.replace(" ", "_").toUpperCase(),
        outcomeDisposition: data.outcomeDisposition
          ? data.outcomeDisposition.replace(/\s+/g, "_").toUpperCase()
          : undefined,
        createdBy: "system",
        updatedBy: "system",
      } as any,
    });

    const response: ApiResponse<ActivityLog> = {
      success: true,
      data: {
        ...activity,
        activityType: activity.activityType.replace("_", " "),
        outcomeDisposition: activity.outcomeDisposition
          ? activity.outcomeDisposition.replace(/_/g, " ")
          : undefined,
      } as ActivityLog,
      message: "Activity created successfully",
    };

    res.status(201).json(response);
  } catch (error) {
    console.error("Error creating activity:", error);
    res
      .status(500)
      .json({ success: false, error: "Failed to create activity" });
  }
};

export const updateActivity: RequestHandler = async (req, res) => {
  try {
    const data: UpdateActivityRequest = req.body;

    const activity = await prisma.activityLog.update({
      where: { id: req.params.id },
      data: {
        ...data,
        activityType: data.activityType
          ? data.activityType.replace(" ", "_").toUpperCase()
          : undefined,
        outcomeDisposition: data.outcomeDisposition
          ? data.outcomeDisposition.replace(/\s+/g, "_").toUpperCase()
          : undefined,
        updatedBy: "system",
      } as any,
    });

    const response: ApiResponse<ActivityLog> = {
      success: true,
      data: {
        ...activity,
        activityType: activity.activityType.replace("_", " "),
        outcomeDisposition: activity.outcomeDisposition
          ? activity.outcomeDisposition.replace(/_/g, " ")
          : undefined,
      } as ActivityLog,
      message: "Activity updated successfully",
    };

    res.json(response);
  } catch (error) {
    console.error("Error updating activity:", error);
    res
      .status(500)
      .json({ success: false, error: "Failed to update activity" });
  }
};

export const deleteActivity: RequestHandler = async (req, res) => {
  try {
    await prisma.activityLog.delete({
      where: { id: req.params.id },
    });

    res.json({ success: true, message: "Activity deleted successfully" });
  } catch (error) {
    console.error("Error deleting activity:", error);
    res
      .status(500)
      .json({ success: false, error: "Failed to delete activity" });
  }
};

// Deals Routes
export const getDeals: RequestHandler = async (req, res) => {
  try {
    const page = parseInt(req.query.page as string) || 1;
    const limit = parseInt(req.query.limit as string) || 10;

    const skip = (page - 1) * limit;

    const [deals, total] = await Promise.all([
      prisma.activeDeal.findMany({
        skip,
        take: limit,
        orderBy: { createdAt: "desc" },
        include: {
          account: true,
          contact: true,
        },
      }),
      prisma.activeDeal.count(),
    ]);

    const response: PaginatedResponse<ActiveDeal> = {
      success: true,
      data: deals.map((deal) => ({
        ...deal,
        businessLine: deal.businessLine
          ? deal.businessLine.replace(/_/g, " ")
          : undefined,
        geo: deal.geo ? deal.geo.replace("_", " ") : undefined,
        entity: deal.entity ? deal.entity.replace("_", " ") : undefined,
        stage: deal.stage ? deal.stage.replace(/_/g, " ") : undefined,
      })) as ActiveDeal[],
      pagination: {
        page,
        limit,
        total,
        totalPages: Math.ceil(total / limit),
      },
    };

    res.json(response);
  } catch (error) {
    console.error("Error fetching deals:", error);
    res.status(500).json({ success: false, error: "Failed to fetch deals" });
  }
};

export const getDeal: RequestHandler = async (req, res) => {
  try {
    const deal = await prisma.activeDeal.findUnique({
      where: { id: req.params.id },
      include: {
        account: true,
        contact: true,
      },
    });

    if (!deal) {
      return res
        .status(404)
        .json({ success: false, message: "Deal not found" });
    }

    const response: ApiResponse<ActiveDeal> = {
      success: true,
      data: {
        ...deal,
        businessLine: deal.businessLine
          ? deal.businessLine.replace(/_/g, " ")
          : undefined,
        geo: deal.geo ? deal.geo.replace("_", " ") : undefined,
        entity: deal.entity ? deal.entity.replace("_", " ") : undefined,
        stage: deal.stage ? deal.stage.replace(/_/g, " ") : undefined,
      } as ActiveDeal,
    };

    res.json(response);
  } catch (error) {
    console.error("Error fetching deal:", error);
    res.status(500).json({ success: false, error: "Failed to fetch deal" });
  }
};

export const createDeal: RequestHandler = async (req, res) => {
  try {
    const data: CreateDealRequest = req.body;

    const deal = await prisma.activeDeal.create({
      data: {
        ...data,
        businessLine: data.businessLine
          ? data.businessLine.replace(/\s+/g, "_").toUpperCase()
          : undefined,
        geo: data.geo ? data.geo.replace(" ", "_").toUpperCase() : undefined,
        entity: data.entity
          ? data.entity.replace(/\s+/g, "_").toUpperCase()
          : undefined,
        stage: data.stage
          ? data.stage.replace(/\s+/g, "_").toUpperCase()
          : undefined,
        createdBy: "system",
        updatedBy: "system",
      } as any,
    });

    const response: ApiResponse<ActiveDeal> = {
      success: true,
      data: {
        ...deal,
        businessLine: deal.businessLine
          ? deal.businessLine.replace(/_/g, " ")
          : undefined,
        geo: deal.geo ? deal.geo.replace("_", " ") : undefined,
        entity: deal.entity ? deal.entity.replace("_", " ") : undefined,
        stage: deal.stage ? deal.stage.replace(/_/g, " ") : undefined,
      } as ActiveDeal,
      message: "Deal created successfully",
    };

    res.status(201).json(response);
  } catch (error) {
    console.error("Error creating deal:", error);
    res.status(500).json({ success: false, error: "Failed to create deal" });
  }
};

export const updateDeal: RequestHandler = async (req, res) => {
  try {
    const data: UpdateDealRequest = req.body;

    const deal = await prisma.activeDeal.update({
      where: { id: req.params.id },
      data: {
        ...data,
        businessLine: data.businessLine
          ? data.businessLine.replace(/\s+/g, "_").toUpperCase()
          : undefined,
        geo: data.geo ? data.geo.replace(" ", "_").toUpperCase() : undefined,
        entity: data.entity
          ? data.entity.replace(/\s+/g, "_").toUpperCase()
          : undefined,
        stage: data.stage
          ? data.stage.replace(/\s+/g, "_").toUpperCase()
          : undefined,
        updatedBy: "system",
      } as any,
    });

    const response: ApiResponse<ActiveDeal> = {
      success: true,
      data: {
        ...deal,
        businessLine: deal.businessLine
          ? deal.businessLine.replace(/_/g, " ")
          : undefined,
        geo: deal.geo ? deal.geo.replace("_", " ") : undefined,
        entity: deal.entity ? deal.entity.replace("_", " ") : undefined,
        stage: deal.stage ? deal.stage.replace(/_/g, " ") : undefined,
      } as ActiveDeal,
      message: "Deal updated successfully",
    };

    res.json(response);
  } catch (error) {
    console.error("Error updating deal:", error);
    res.status(500).json({ success: false, error: "Failed to update deal" });
  }
};

export const deleteDeal: RequestHandler = async (req, res) => {
  try {
    await prisma.activeDeal.delete({
      where: { id: req.params.id },
    });

    res.json({ success: true, message: "Deal deleted successfully" });
  } catch (error) {
    console.error("Error deleting deal:", error);
    res.status(500).json({ success: false, error: "Failed to delete deal" });
  }
};

// Leads Routes
export const getLeads: RequestHandler = async (req, res) => {
  try {
    const page = parseInt(req.query.page as string) || 1;
    const limit = parseInt(req.query.limit as string) || 10;

    const skip = (page - 1) * limit;

    const [leads, total] = await Promise.all([
      prisma.lead.findMany({
        skip,
        take: limit,
        orderBy: { createdAt: "desc" },
      }),
      prisma.lead.count(),
    ]);

    const response: PaginatedResponse<Lead> = {
      success: true,
      data: leads.map((lead) => ({
        ...lead,
        leadSource: lead.leadSource
          ? lead.leadSource.replace(/_/g, " ")
          : undefined,
        status: lead.status ? lead.status.replace("_", " ") : undefined,
        rating: lead.rating ? lead.rating.replace("_", " ") : undefined,
      })) as Lead[],
      pagination: {
        page,
        limit,
        total,
        totalPages: Math.ceil(total / limit),
      },
    };

    res.json(response);
  } catch (error) {
    console.error("Error fetching leads:", error);
    res.status(500).json({ success: false, error: "Failed to fetch leads" });
  }
};

export const getLead: RequestHandler = async (req, res) => {
  try {
    const lead = await prisma.lead.findUnique({
      where: { id: req.params.id },
    });

    if (!lead) {
      return res
        .status(404)
        .json({ success: false, message: "Lead not found" });
    }

    const response: ApiResponse<Lead> = {
      success: true,
      data: {
        ...lead,
        leadSource: lead.leadSource
          ? lead.leadSource.replace(/_/g, " ")
          : undefined,
        status: lead.status ? lead.status.replace("_", " ") : undefined,
        rating: lead.rating ? lead.rating.replace("_", " ") : undefined,
      } as Lead,
    };

    res.json(response);
  } catch (error) {
    console.error("Error fetching lead:", error);
    res.status(500).json({ success: false, error: "Failed to fetch lead" });
  }
};

export const createLead: RequestHandler = async (req, res) => {
  try {
    const data: CreateLeadRequest = req.body;

    const lead = await prisma.lead.create({
      data: {
        ...data,
        leadSource: data.leadSource
          ? data.leadSource.replace(/\s+/g, "_").toUpperCase()
          : undefined,
        status: data.status
          ? data.status.replace(" ", "_").toUpperCase()
          : undefined,
        rating: data.rating
          ? data.rating.replace(" ", "_").toUpperCase()
          : undefined,
        createdBy: "system",
        updatedBy: "system",
      } as any,
    });

    const response: ApiResponse<Lead> = {
      success: true,
      data: {
        ...lead,
        leadSource: lead.leadSource
          ? lead.leadSource.replace(/_/g, " ")
          : undefined,
        status: lead.status ? lead.status.replace("_", " ") : undefined,
        rating: lead.rating ? lead.rating.replace("_", " ") : undefined,
      } as Lead,
      message: "Lead created successfully",
    };

    res.status(201).json(response);
  } catch (error) {
    console.error("Error creating lead:", error);
    res.status(500).json({ success: false, error: "Failed to create lead" });
  }
};

export const updateLead: RequestHandler = async (req, res) => {
  try {
    const data: UpdateLeadRequest = req.body;

    const lead = await prisma.lead.update({
      where: { id: req.params.id },
      data: {
        ...data,
        leadSource: data.leadSource
          ? data.leadSource.replace(/\s+/g, "_").toUpperCase()
          : undefined,
        status: data.status
          ? data.status.replace(" ", "_").toUpperCase()
          : undefined,
        rating: data.rating
          ? data.rating.replace(" ", "_").toUpperCase()
          : undefined,
        updatedBy: "system",
      } as any,
    });

    const response: ApiResponse<Lead> = {
      success: true,
      data: {
        ...lead,
        leadSource: lead.leadSource
          ? lead.leadSource.replace(/_/g, " ")
          : undefined,
        status: lead.status ? lead.status.replace("_", " ") : undefined,
        rating: lead.rating ? lead.rating.replace("_", " ") : undefined,
      } as Lead,
      message: "Lead updated successfully",
    };

    res.json(response);
  } catch (error) {
    console.error("Error updating lead:", error);
    res.status(500).json({ success: false, error: "Failed to update lead" });
  }
};

export const deleteLead: RequestHandler = async (req, res) => {
  try {
    await prisma.lead.delete({
      where: { id: req.params.id },
    });

    res.json({ success: true, message: "Lead deleted successfully" });
  } catch (error) {
    console.error("Error deleting lead:", error);
    res.status(500).json({ success: false, error: "Failed to delete lead" });
  }
};

// Reports Routes
export const generateReport: RequestHandler = async (req, res) => {
  try {
    const data: ReportRequest = req.body;

    // For now, return a simple report based on current data
    const contacts = await prisma.contact.count();
    const accounts = await prisma.account.count();
    const deals = await prisma.activeDeal.count();
    const activities = await prisma.activityLog.count();
    const leads = await prisma.lead.count();

    const reportData = {
      id: `report_${Date.now()}`,
      reportType: data.reportType,
      period: data.period,
      salesRep: data.salesRep,
      geo: data.geo,
      businessLine: data.businessLine,
      metrics: {
        totalContacts: contacts,
        totalAccounts: accounts,
        totalDeals: deals,
        totalActivities: activities,
        totalLeads: leads,
      },
      generatedAt: new Date(),
      generatedBy: "system",
    };

    res.json({
      success: true,
      data: reportData,
      message: "Report generated successfully",
    });
  } catch (error) {
    console.error("Error generating report:", error);
    res
      .status(500)
      .json({ success: false, error: "Failed to generate report" });
  }
};

export const downloadReport: RequestHandler = async (req, res) => {
  try {
    // For now, return a simple download response
    res.json({ success: true, message: "Report download feature coming soon" });
  } catch (error) {
    console.error("Error downloading report:", error);
    res
      .status(500)
      .json({ success: false, error: "Failed to download report" });
  }
};

// User Profile Routes
export const getUserProfile: RequestHandler = async (req, res) => {
  try {
    const userId = req.params.id;
    const userProfile = await prisma.userProfile.findUnique({
      where: { id: userId },
    });

    if (!userProfile) {
      return res
        .status(404)
        .json({ success: false, message: "User profile not found" });
    }

    const response: ApiResponse<UserProfile> = {
      success: true,
      data: {
        ...userProfile,
        role: userProfile.role?.replace(/_/g, " "),
        notifications: {
          emailNotifications: userProfile.emailNotifications,
          smsNotifications: userProfile.smsNotifications,
          pushNotifications: userProfile.pushNotifications,
        },
      } as UserProfile,
    };

    res.json(response);
  } catch (error) {
    console.error("Error fetching user profile:", error);
    res
      .status(500)
      .json({ success: false, error: "Failed to fetch user profile" });
  }
};

export const getCurrentUserProfile: RequestHandler = async (req, res) => {
  try {
    // For now, return the first user profile or create a default one
    let userProfile = await prisma.userProfile.findFirst();

    if (!userProfile) {
      // Create default user profile
      userProfile = await prisma.userProfile.create({
        data: {
          firstName: "John",
          lastName: "Doe",
          email: "john.doe@yitro.com",
          title: "Sales Manager",
          department: "Sales",
          role: "SALES_MANAGER",
          phone: "+1-555-0100",
          timezone: "EST",
          language: "en",
        },
      });
    }

    const response: ApiResponse<UserProfile> = {
      success: true,
      data: {
        ...userProfile,
        role: userProfile.role?.replace(/_/g, " "),
        notifications: {
          emailNotifications: userProfile.emailNotifications,
          smsNotifications: userProfile.smsNotifications,
          pushNotifications: userProfile.pushNotifications,
        },
      } as UserProfile,
    };

    res.json(response);
  } catch (error) {
    console.error("Error fetching current user profile:", error);
    res
      .status(500)
      .json({ success: false, error: "Failed to fetch user profile" });
  }
};

export const updateUserProfile: RequestHandler = async (req, res) => {
  try {
    const userId = req.params.id;
    const data: UpdateUserProfileRequest = req.body;

    const userProfile = await prisma.userProfile.update({
      where: { id: userId },
      data: {
        ...data,
        role: data.role
          ? data.role.replace(/\s+/g, "_").toUpperCase()
          : undefined,
        emailNotifications: data.notifications?.emailNotifications,
        smsNotifications: data.notifications?.smsNotifications,
        pushNotifications: data.notifications?.pushNotifications,
      } as any,
    });

    const response: ApiResponse<UserProfile> = {
      success: true,
      data: {
        ...userProfile,
        role: userProfile.role?.replace(/_/g, " "),
        notifications: {
          emailNotifications: userProfile.emailNotifications,
          smsNotifications: userProfile.smsNotifications,
          pushNotifications: userProfile.pushNotifications,
        },
      } as UserProfile,
      message: "Profile updated successfully",
    };

    res.json(response);
  } catch (error) {
    console.error("Error updating user profile:", error);
    res
      .status(500)
      .json({ success: false, error: "Failed to update user profile" });
  }
};

export const updateCurrentUserProfile: RequestHandler = async (req, res) => {
  try {
    const data: UpdateUserProfileRequest = req.body;

    // Update the first (current) user profile
    let userProfile = await prisma.userProfile.findFirst();

    if (!userProfile) {
      return res
        .status(404)
        .json({ success: false, message: "User profile not found" });
    }

    userProfile = await prisma.userProfile.update({
      where: { id: userProfile.id },
      data: {
        ...data,
        role: data.role
          ? data.role.replace(/\s+/g, "_").toUpperCase()
          : undefined,
        emailNotifications: data.notifications?.emailNotifications,
        smsNotifications: data.notifications?.smsNotifications,
        pushNotifications: data.notifications?.pushNotifications,
      } as any,
    });

    const response: ApiResponse<UserProfile> = {
      success: true,
      data: {
        ...userProfile,
        role: userProfile.role?.replace(/_/g, " "),
        notifications: {
          emailNotifications: userProfile.emailNotifications,
          smsNotifications: userProfile.smsNotifications,
          pushNotifications: userProfile.pushNotifications,
        },
      } as UserProfile,
      message: "Profile updated successfully",
    };

    res.json(response);
  } catch (error) {
    console.error("Error updating user profile:", error);
    res
      .status(500)
      .json({ success: false, error: "Failed to update user profile" });
  }
};

// Initialize sample data
export const initializeSampleData = async () => {
  try {
    // Check if we already have comprehensive sample data
    const [contactCount, dealCount, leadCount] = await Promise.all([
      prisma.contact.count(),
      prisma.activeDeal.count(),
      prisma.lead.count(),
    ]);

    if (contactCount > 0 && dealCount > 2 && leadCount > 0) {
      console.log(
        "Comprehensive sample data already exists, skipping initialization",
      );
      return;
    }

    console.log("Initializing or updating sample data...");

    console.log("Initializing sample data...");

    // Create sample accounts first
    const techCorpAccount = await prisma.account.create({
      data: {
        accountName: "TechCorp Solutions",
        accountRating: "GOLD",
        accountOwner: "Sales Rep 1",
        status: "PROSPECT",
        industry: "Technology",
        revenue: "$10M - $50M",
        numberOfEmployees: "100-500",
        addressLine1: "123 Tech Street",
        city: "New York",
        state: "NY",
        country: "USA",
        zipPostCode: "10001",
        timeZone: "EST",
        website: "https://techcorp.com",
        geo: "AMERICAS",
        createdBy: "system",
        updatedBy: "system",
      },
    });

    const innovateAccount = await prisma.account.create({
      data: {
        accountName: "Innovate Inc",
        accountRating: "PLATINUM",
        accountOwner: "Sales Rep 2",
        status: "ACTIVE_DEAL",
        industry: "Software",
        revenue: "$50M+",
        numberOfEmployees: "500+",
        addressLine1: "456 Innovation Ave",
        city: "San Francisco",
        state: "CA",
        country: "USA",
        zipPostCode: "94105",
        timeZone: "PST",
        website: "https://innovate.com",
        geo: "AMERICAS",
        createdBy: "system",
        updatedBy: "system",
      },
    });

    // Create sample contacts with proper account references
    const johnContact = await prisma.contact.create({
      data: {
        firstName: "John",
        lastName: "Smith",
        title: "CEO",
        associatedAccount: techCorpAccount.id,
        emailAddress: "john.smith@techcorp.com",
        deskPhone: "+1-555-0123",
        mobilePhone: "+1-555-0124",
        city: "New York",
        state: "NY",
        country: "USA",
        timeZone: "EST",
        source: "DATA_RESEARCH",
        owner: "Sales Rep 1",
        status: "PROSPECT",
        createdBy: "system",
        updatedBy: "system",
      },
    });

    const janeContact = await prisma.contact.create({
      data: {
        firstName: "Jane",
        lastName: "Doe",
        title: "CTO",
        associatedAccount: innovateAccount.id,
        emailAddress: "jane.doe@innovate.com",
        deskPhone: "+1-555-0125",
        mobilePhone: "+1-555-0126",
        city: "San Francisco",
        state: "CA",
        country: "USA",
        timeZone: "PST",
        source: "REFERRAL",
        owner: "Sales Rep 2",
        status: "ACTIVE_DEAL",
        createdBy: "system",
        updatedBy: "system",
      },
    });

    // Create sample leads
    await prisma.lead.createMany({
      data: [
        {
          firstName: "Alice",
          lastName: "Johnson",
          company: "StartupCorp",
          title: "Founder",
          email: "alice@startupcorp.com",
          phone: "+1-555-0200",
          leadSource: "WEBSITE",
          status: "NEW",
          rating: "HOT",
          owner: "Sales Rep 1",
          createdBy: "system",
          updatedBy: "system",
        },
        {
          firstName: "Bob",
          lastName: "Wilson",
          company: "Enterprise Ltd",
          title: "CTO",
          email: "bob@enterprise.com",
          phone: "+1-555-0201",
          leadSource: "REFERRAL",
          status: "WORKING",
          rating: "WARM",
          owner: "Sales Rep 2",
          createdBy: "system",
          updatedBy: "system",
        },
      ],
    });

    // Create sample deals with proper account and contact references
    await prisma.activeDeal.createMany({
      data: [
        {
          dealName: "TechCorp Automation Project",
          dealOwner: "Sales Rep 1",
          businessLine: "AUTOMATION",
          associatedAccount: techCorpAccount.id,
          associatedContact: johnContact.id,
          dealValue: "150000",
          stage: "PROPOSAL_SUBMITTED",
          probability: "75%",
          description: "Large automation project for TechCorp",
          geo: "AMERICAS",
          entity: "YITRO_GLOBAL",
          createdBy: "system",
          updatedBy: "system",
        },
        {
          dealName: "Innovate Software Development",
          dealOwner: "Sales Rep 2",
          businessLine: "PRODUCT",
          associatedAccount: innovateAccount.id,
          associatedContact: janeContact.id,
          dealValue: "250000",
          stage: "NEGOTIATING",
          probability: "90%",
          description: "Custom software development for Innovate Inc",
          geo: "AMERICAS",
          entity: "YITRO_TECH",
          createdBy: "system",
          updatedBy: "system",
        },
        {
          dealName: "Enterprise Analytics Platform",
          dealOwner: "Sales Rep 1",
          businessLine: "SOLUTION",
          associatedAccount: techCorpAccount.id,
          dealValue: "500000",
          stage: "ORDER_WON",
          probability: "100%",
          description: "Complete analytics solution for large enterprise",
          geo: "AMERICAS",
          entity: "YITRO_GLOBAL",
          createdBy: "system",
          updatedBy: "system",
        },
        {
          dealName: "Startup CRM Integration",
          dealOwner: "Sales Rep 2",
          businessLine: "AUTOMATION",
          associatedAccount: innovateAccount.id,
          dealValue: "75000",
          stage: "ORDER_WON",
          probability: "100%",
          description: "CRM integration for growing startup",
          geo: "AMERICAS",
          entity: "YITRO_TECH",
          createdBy: "system",
          updatedBy: "system",
        },
      ],
    });

    // Create sample activities with proper account and contact references
    await prisma.activityLog.createMany({
      data: [
        {
          activityType: "CALL",
          dateTime: new Date("2024-01-07T10:00:00Z"),
          summary:
            "Initial discovery call with TechCorp. Discussed automation needs and project scope.",
          outcomeDisposition: "MEETING_COMPLETED",
          associatedAccount: techCorpAccount.id,
          associatedContact: johnContact.id,
          createdBy: "system",
          updatedBy: "system",
        },
        {
          activityType: "EMAIL",
          dateTime: new Date("2024-01-07T14:30:00Z"),
          summary:
            "Sent proposal document and technical specifications to Innovate Inc.",
          outcomeDisposition: "EMAIL_SENT",
          associatedAccount: innovateAccount.id,
          associatedContact: janeContact.id,
          createdBy: "system",
          updatedBy: "system",
        },
        {
          activityType: "LINKEDIN_MSG",
          dateTime: new Date("2024-01-06T16:45:00Z"),
          summary:
            "Connected with potential lead on LinkedIn and shared product demo.",
          outcomeDisposition: "CALLBACK_REQUESTED",
          createdBy: "system",
          updatedBy: "system",
        },
      ],
    });

    console.log("Sample data initialized successfully");
  } catch (error) {
    console.error("Error initializing sample data:", error);
  }
};
