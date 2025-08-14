import { useState, useEffect } from "react";
import { useCRM } from "../contexts/CRMContext";

interface Lead {
  id: number;
  name: string;
  company: string;
  title: string;
  email: string;
  phone: string;
  status: string;
  source: string;
  score: number;
  value: string;
  lastActivity: string;
}

interface Account {
  id: number;
  name: string;
  industry: string;
  type: string;
  revenue: string;
  employees: string;
  location: string;
  phone: string;
  website: string;
  owner: string;
  rating: string;
  lastActivity: string;
  activeDeals: number;
  contacts: number;
}

interface Contact {
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
  source?: "Data Research" | "Referral" | "Event";
  owner?: string;
  ownerId?: string;
  status?: "Suspect" | "Prospect" | "Active Deal" | "Do Not Call";
  createdAt: string;
  updatedAt: string;
}

interface ActiveDeal {
  id: string;
  dealName: string;
  businessLine: string;
  associatedAccount: string;
  associatedContact: string;
  closingDate: string;
  probability: number;
  dealValue: number;
  approvedBy: string;
  description: string;
  nextStep: string;
  geo: string;
  entity: string;
  stage: string;
  owner: string;
  ownerId: string;
  createdAt: string;
}

interface CRMMetrics {
  totalRevenue: number;
  activeDeals: number;
  newLeads: number;
  closedDeals: number;
  myAccounts: number;
  myContacts: number;
  totalLeads: number;
  totalAccounts: number;
  totalContacts: number;
  totalDeals: number;
  avgDealValue: number;
  conversionRate: number;
}

export function useCRMData() {
  const { leads, accounts, contacts, deals } = useCRM();
  const [loading, setLoading] = useState(true);
  const [crmMetrics, setCrmMetrics] = useState<CRMMetrics>({
    totalRevenue: 0,
    activeDeals: 0,
    newLeads: 0,
    closedDeals: 0,
    myAccounts: 0,
    myContacts: 0,
    totalLeads: 0,
    totalAccounts: 0,
    totalContacts: 0,
    totalDeals: 0,
    avgDealValue: 0,
    conversionRate: 0,
  });

  useEffect(() => {
    calculateCRMMetrics();
  }, [leads, accounts, contacts, deals]);

  const calculateCRMMetrics = async () => {
    try {
      setLoading(true);

      // Simulate calculation delay
      await new Promise((resolve) => setTimeout(resolve, 500));

      // Calculate metrics from real CRM context data
      const totalLeads = leads.length;
      const newLeads = leads.filter((lead) => lead.status === "New").length;

      const totalAccounts = accounts.length;
      const myAccounts = totalAccounts; // Assuming current user sees all for demo

      const totalContacts = contacts.length;
      const myContacts = totalContacts; // Assuming current user sees all for demo

      const totalDeals = deals.length;
      const activeDeals = deals.filter(
        (deal) => !["Order Won", "Order Lost"].includes(deal.stage),
      ).length;
      const closedDeals = deals.filter(
        (deal) => deal.stage === "Order Won",
      ).length;

      const totalDealValue = deals.reduce(
        (sum, deal) => sum + deal.dealValue,
        0,
      );
      const totalRevenue = deals
        .filter((deal) => deal.stage === "Order Won")
        .reduce((sum, deal) => sum + deal.dealValue, 0);

      const avgDealValue = totalDeals > 0 ? totalDealValue / totalDeals : 0;
      const conversionRate =
        totalLeads > 0 ? (closedDeals / totalLeads) * 100 : 0;

      const metrics: CRMMetrics = {
        totalRevenue,
        activeDeals,
        newLeads,
        closedDeals,
        myAccounts,
        myContacts,
        totalLeads,
        totalAccounts,
        totalContacts,
        totalDeals,
        avgDealValue,
        conversionRate,
      };

      setCrmMetrics(metrics);
    } catch (error) {
      console.error("Failed to fetch CRM data:", error);
    } finally {
      setLoading(false);
    }
  };

  return {
    crmMetrics,
    loading,
    refreshData: calculateCRMMetrics,
  };
}
