import React, {
  createContext,
  useContext,
  useState,
  ReactNode,
  useEffect,
} from "react";

// Types
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

interface CRMContextType {
  // Data
  leads: Lead[];
  accounts: Account[];
  contacts: Contact[];
  deals: ActiveDeal[];

  // Lead operations
  addLead: (lead: Omit<Lead, "id">) => void;
  updateLead: (id: number, updates: Partial<Lead>) => void;
  deleteLead: (id: number) => void;

  // Account operations
  addAccount: (account: Omit<Account, "id">) => void;
  updateAccount: (id: number, updates: Partial<Account>) => void;
  deleteAccount: (id: number) => void;

  // Contact operations
  addContact: (contact: Omit<Contact, "id">) => void;
  updateContact: (id: string, updates: Partial<Contact>) => void;
  deleteContact: (id: string) => void;

  // Deal operations
  addDeal: (deal: Omit<ActiveDeal, "id">) => void;
  updateDeal: (id: string, updates: Partial<ActiveDeal>) => void;
  deleteDeal: (id: string) => void;
}

const CRMContext = createContext<CRMContextType | undefined>(undefined);

// Initial data
const initialLeads: Lead[] = [
  {
    id: 1,
    name: "Alice Johnson",
    company: "StartupCorp",
    title: "CEO",
    email: "alice@startupcorp.com",
    phone: "+1 (555) 123-4567",
    status: "New",
    source: "Website",
    score: 85,
    value: "$50,000",
    lastActivity: "2 hours ago",
  },
  {
    id: 2,
    name: "Bob Wilson",
    company: "Enterprise Ltd",
    title: "CTO",
    email: "bob@enterprise.com",
    phone: "+1 (555) 987-6543",
    status: "Qualified",
    source: "Referral",
    score: 92,
    value: "$125,000",
    lastActivity: "1 day ago",
  },
  {
    id: 3,
    name: "Carol Davis",
    company: "Tech Innovations",
    title: "VP Sales",
    email: "carol@techinnovations.com",
    phone: "+1 (555) 456-7890",
    status: "Working",
    source: "Cold Call",
    score: 78,
    value: "$75,000",
    lastActivity: "3 hours ago",
  },
  {
    id: 4,
    name: "David Brown",
    company: "Future Systems",
    title: "Director",
    email: "david@futuresystems.com",
    phone: "+1 (555) 321-0987",
    status: "Nurturing",
    source: "LinkedIn",
    score: 65,
    value: "$30,000",
    lastActivity: "5 days ago",
  },
];

const initialAccounts: Account[] = [
  {
    id: 1,
    name: "TechCorp Solutions",
    industry: "Technology",
    type: "Customer",
    revenue: "$2.5M",
    employees: "100-500",
    location: "New York, NY",
    phone: "+1 (555) 123-4567",
    website: "techcorp.com",
    owner: "John Smith",
    rating: "Hot",
    lastActivity: "2 days ago",
    activeDeals: 3,
    contacts: 8,
  },
  {
    id: 2,
    name: "Innovate Inc",
    industry: "Software",
    type: "Prospect",
    revenue: "$5M+",
    employees: "500+",
    location: "San Francisco, CA",
    phone: "+1 (555) 987-6543",
    website: "innovate.com",
    owner: "Jane Doe",
    rating: "Warm",
    lastActivity: "1 week ago",
    activeDeals: 1,
    contacts: 5,
  },
  {
    id: 3,
    name: "StartupTech",
    industry: "Fintech",
    type: "Customer",
    revenue: "$500K",
    employees: "10-50",
    location: "Austin, TX",
    phone: "+1 (555) 456-7890",
    website: "startuptech.io",
    owner: "Mike Johnson",
    rating: "Cold",
    lastActivity: "3 days ago",
    activeDeals: 2,
    contacts: 3,
  },
  {
    id: 4,
    name: "Global Industries",
    industry: "Manufacturing",
    type: "Partner",
    revenue: "$50M+",
    employees: "1000+",
    location: "Chicago, IL",
    phone: "+1 (555) 321-0987",
    website: "globalind.com",
    owner: "Sarah Wilson",
    rating: "Hot",
    lastActivity: "Yesterday",
    activeDeals: 5,
    contacts: 12,
  },
];

const initialContacts: Contact[] = [
  {
    id: "contact-001",
    firstName: "John",
    lastName: "Smith",
    title: "CTO",
    associatedAccount: "TechCorp Solutions",
    emailAddress: "john.smith@techcorp.com",
    deskPhone: "+1 (555) 123-4567",
    mobilePhone: "+1 (555) 123-4568",
    city: "New York",
    state: "NY",
    country: "United States",
    timeZone: "EST",
    source: "Data Research",
    owner: "Jane Doe",
    ownerId: "user-001",
    status: "Active Deal",
    createdAt: "2024-01-10",
    updatedAt: "2024-02-15",
  },
  {
    id: "contact-002",
    firstName: "Sarah",
    lastName: "Wilson",
    title: "VP of Operations",
    associatedAccount: "Innovate Inc",
    emailAddress: "sarah.wilson@innovate.com",
    deskPhone: "+1 (555) 987-6543",
    mobilePhone: "+1 (555) 987-6544",
    city: "San Francisco",
    state: "CA",
    country: "United States",
    timeZone: "PST",
    source: "Referral",
    owner: "Mike Johnson",
    ownerId: "user-002",
    status: "Prospect",
    createdAt: "2024-01-15",
    updatedAt: "2024-02-10",
  },
  {
    id: "contact-003",
    firstName: "Michael",
    lastName: "Chen",
    title: "Director of IT",
    associatedAccount: "StartupTech",
    emailAddress: "michael.chen@startuptech.io",
    deskPhone: "+1 (555) 456-7890",
    mobilePhone: "+1 (555) 456-7891",
    city: "Austin",
    state: "TX",
    country: "United States",
    timeZone: "CST",
    source: "Event",
    owner: "Sarah Wilson",
    ownerId: "user-003",
    status: "Prospect",
    createdAt: "2024-01-20",
    updatedAt: "2024-02-12",
  },
  {
    id: "contact-004",
    firstName: "Lisa",
    lastName: "Garcia",
    title: "CEO",
    associatedAccount: "Global Industries",
    emailAddress: "lisa.garcia@globalind.com",
    deskPhone: "+1 (555) 321-0987",
    mobilePhone: "+1 (555) 321-0988",
    city: "Chicago",
    state: "IL",
    country: "United States",
    timeZone: "CST",
    source: "Data Research",
    owner: "Alex Chen",
    ownerId: "user-004",
    status: "Active Deal",
    createdAt: "2024-01-25",
    updatedAt: "2024-02-18",
  },
  {
    id: "contact-005",
    firstName: "David",
    lastName: "Brown",
    title: "Product Manager",
    associatedAccount: "MegaCorp International",
    emailAddress: "david.brown@megacorp.com",
    deskPhone: "+1 (555) 654-3210",
    mobilePhone: "+1 (555) 654-3211",
    city: "Boston",
    state: "MA",
    country: "United States",
    timeZone: "EST",
    source: "Referral",
    owner: "Dr. Patel",
    ownerId: "user-005",
    status: "Suspect",
    createdAt: "2024-02-01",
    updatedAt: "2024-02-20",
  },
];

const initialDeals: ActiveDeal[] = [
  {
    id: "deal-001",
    dealName: "Enterprise Software Package",
    businessLine: "Human Capital",
    associatedAccount: "TechCorp Solutions",
    associatedContact: "John Smith",
    closingDate: "2024-03-15",
    probability: 75,
    dealValue: 125000,
    approvedBy: "Sarah Wilson",
    description: "Comprehensive HR management solution",
    nextStep: "Final contract review",
    geo: "Americas",
    entity: "Yitro Global",
    stage: "Negotiating",
    owner: "Jane Doe",
    ownerId: "user-001",
    createdAt: "2024-01-10",
  },
  {
    id: "deal-002",
    dealName: "Cloud Migration Services",
    businessLine: "Managed Services",
    associatedAccount: "Innovate Inc",
    associatedContact: "Mike Johnson",
    closingDate: "2024-02-28",
    probability: 60,
    dealValue: 85000,
    approvedBy: "David Brown",
    description: "Complete cloud infrastructure migration",
    nextStep: "Technical proposal submission",
    geo: "Americas",
    entity: "Yitro Tech",
    stage: "Proposal Submitted",
    owner: "John Smith",
    ownerId: "user-002",
    createdAt: "2024-01-15",
  },
  {
    id: "deal-003",
    dealName: "GCC Automation Platform",
    businessLine: "Automation",
    associatedAccount: "Global Industries",
    associatedContact: "Sarah Wilson",
    closingDate: "2024-04-20",
    probability: 90,
    dealValue: 250000,
    approvedBy: "Jennifer Lee",
    description: "Advanced process automation solution",
    nextStep: "Implementation planning",
    geo: "EMEA",
    entity: "Yitro Global",
    stage: "Closing",
    owner: "Mike Johnson",
    ownerId: "user-003",
    createdAt: "2024-01-05",
  },
  {
    id: "deal-004",
    dealName: "Support Package Renewal",
    businessLine: "Support",
    associatedAccount: "StartupTech",
    associatedContact: "Alex Chen",
    closingDate: "2024-03-30",
    probability: 95,
    dealValue: 45000,
    approvedBy: "Robert Kim",
    description: "Annual support and maintenance renewal",
    nextStep: "Contract signing",
    geo: "India",
    entity: "Yitro Support",
    stage: "Order Won",
    owner: "Sarah Wilson",
    ownerId: "user-004",
    createdAt: "2024-02-01",
  },
  {
    id: "deal-005",
    dealName: "Product Solution Implementation",
    businessLine: "Product",
    associatedAccount: "MegaCorp International",
    associatedContact: "Lisa Garcia",
    closingDate: "2024-05-15",
    probability: 45,
    dealValue: 180000,
    approvedBy: "Tom Anderson",
    description: "Custom product development and implementation",
    nextStep: "Requirements gathering",
    geo: "Philippines",
    entity: "Yitro Global",
    stage: "Opportunity Identified",
    owner: "Alex Chen",
    ownerId: "user-005",
    createdAt: "2024-02-10",
  },
  {
    id: "deal-006",
    dealName: "RCM Platform Integration",
    businessLine: "RCM",
    associatedAccount: "HealthSystem Plus",
    associatedContact: "Dr. Maria Rodriguez",
    closingDate: "2024-04-05",
    probability: 70,
    dealValue: 95000,
    approvedBy: "Chris Taylor",
    description: "Revenue cycle management platform integration",
    nextStep: "Demo presentation",
    geo: "ANZ",
    entity: "Yitro Health",
    stage: "Negotiating",
    owner: "Dr. Patel",
    ownerId: "user-006",
    createdAt: "2024-01-20",
  },
];

// Helper function to load data from localStorage with fallback
const loadFromLocalStorage = <T,>(key: string, fallback: T): T => {
  try {
    const saved = localStorage.getItem(key);
    if (saved) {
      const parsed = JSON.parse(saved);
      console.log(
        `📂 Loaded ${key} from localStorage:`,
        parsed.length,
        "items",
      );
      return parsed;
    } else {
      console.log(
        `📂 No saved data for ${key}, using initial data:`,
        fallback.length,
        "items",
      );
    }
  } catch (error) {
    console.error(`Error loading ${key} from localStorage:`, error);
  }
  return fallback;
};

export function CRMProvider({ children }: { children: ReactNode }) {
  const initTime = new Date().toISOString();
  console.log("🏗️ CRMProvider initializing at:", initTime);

  // Track provider recreations
  (window as any).crmProviderInitTime = initTime;

  // Initialize state directly from localStorage or fallback to initial data
  const [leads, setLeads] = useState<Lead[]>(() => {
    const data = loadFromLocalStorage("crm-leads", initialLeads);
    console.log("🏗️ Initial leads loaded:", data.length);
    return data;
  });
  const [accounts, setAccounts] = useState<Account[]>(() => {
    const data = loadFromLocalStorage("crm-accounts", initialAccounts);
    console.log("🏗️ Initial accounts loaded:", data.length);
    return data;
  });
  const [contacts, setContacts] = useState<Contact[]>(() => {
    const data = loadFromLocalStorage("crm-contacts", initialContacts);
    console.log("🏗️ Initial contacts loaded:", data.length);
    console.log(
      "🏗️ Contact names:",
      data.map((c) => `${c.firstName} ${c.lastName}`),
    );
    return data;
  });
  const [deals, setDeals] = useState<ActiveDeal[]>(() => {
    const data = loadFromLocalStorage("crm-deals", initialDeals);
    console.log("🏗️ Initial deals loaded:", data.length);
    return data;
  });

  // Persist data to localStorage whenever state changes
  useEffect(() => {
    try {
      localStorage.setItem("crm-leads", JSON.stringify(leads));
    } catch (error) {
      console.error("Error saving leads to localStorage:", error);
    }
  }, [leads]);

  useEffect(() => {
    try {
      localStorage.setItem("crm-accounts", JSON.stringify(accounts));
    } catch (error) {
      console.error("Error saving accounts to localStorage:", error);
    }
  }, [accounts]);

  useEffect(() => {
    try {
      localStorage.setItem("crm-contacts", JSON.stringify(contacts));
      console.log(
        "💾 Contacts saved to localStorage:",
        contacts.length,
        "contacts",
      );
    } catch (error) {
      console.error("Error saving contacts to localStorage:", error);
    }
  }, [contacts]);

  useEffect(() => {
    try {
      localStorage.setItem("crm-deals", JSON.stringify(deals));
    } catch (error) {
      console.error("Error saving deals to localStorage:", error);
    }
  }, [deals]);

  // Lead operations
  const addLead = (leadData: Omit<Lead, "id">) => {
    const newLead: Lead = {
      ...leadData,
      id: Date.now(),
    };
    setLeads((prev) => [...prev, newLead]);
    console.log("Lead added:", newLead);
  };

  const updateLead = (id: number, updates: Partial<Lead>) => {
    setLeads((prev) =>
      prev.map((lead) => (lead.id === id ? { ...lead, ...updates } : lead)),
    );
    console.log("Lead updated:", id, updates);
  };

  const deleteLead = (id: number) => {
    setLeads((prev) => prev.filter((lead) => lead.id !== id));
    console.log("Lead deleted:", id);
  };

  // Account operations
  const addAccount = (accountData: Omit<Account, "id">) => {
    const newAccount: Account = {
      ...accountData,
      id: Date.now(),
    };
    setAccounts((prev) => [...prev, newAccount]);
    console.log("Account added:", newAccount);
  };

  const updateAccount = (id: number, updates: Partial<Account>) => {
    setAccounts((prev) =>
      prev.map((account) =>
        account.id === id ? { ...account, ...updates } : account,
      ),
    );
    console.log("Account updated:", id, updates);
  };

  const deleteAccount = (id: number) => {
    setAccounts((prev) => prev.filter((account) => account.id !== id));
    console.log("Account deleted:", id);
  };

  // Contact operations
  const addContact = (contactData: Omit<Contact, "id">) => {
    const newContact: Contact = {
      ...contactData,
      id: `contact-${Date.now()}`,
    };
    setContacts((prev) => {
      const updated = [...prev, newContact];
      console.log("Contact added:", newContact);
      console.log("Total contacts after add:", updated.length);
      return updated;
    });
  };

  const updateContact = (id: string, updates: Partial<Contact>) => {
    setContacts((prev) => {
      const updated = prev.map((contact) =>
        contact.id === id
          ? { ...contact, ...updates, updatedAt: new Date().toISOString() }
          : contact,
      );
      console.log("Contact updated:", id, updates);
      console.log("Total contacts after update:", updated.length);
      return updated;
    });
  };

  const deleteContact = (id: string) => {
    setContacts((prev) => {
      const updated = prev.filter((contact) => contact.id !== id);
      console.log("Contact deleted:", id);
      console.log("Total contacts after delete:", updated.length);
      return updated;
    });
  };

  // Deal operations
  const addDeal = (dealData: Omit<ActiveDeal, "id">) => {
    const newDeal: ActiveDeal = {
      ...dealData,
      id: `deal-${Date.now()}`,
    };
    setDeals((prev) => [...prev, newDeal]);
    console.log("Deal added:", newDeal);
  };

  const updateDeal = (id: string, updates: Partial<ActiveDeal>) => {
    setDeals((prev) =>
      prev.map((deal) => (deal.id === id ? { ...deal, ...updates } : deal)),
    );
    console.log("Deal updated:", id, updates);
  };

  const deleteDeal = (id: string) => {
    setDeals((prev) => prev.filter((deal) => deal.id !== id));
    console.log("Deal deleted:", id);
  };

  // Debug function to check localStorage (useful for debugging)
  useEffect(() => {
    // Add global debug function
    (window as any).debugCRM = () => {
      console.log("🔍 Current localStorage data:");
      console.log(
        "Leads:",
        JSON.parse(localStorage.getItem("crm-leads") || "[]").length,
      );
      console.log(
        "Accounts:",
        JSON.parse(localStorage.getItem("crm-accounts") || "[]").length,
      );
      console.log(
        "Contacts:",
        JSON.parse(localStorage.getItem("crm-contacts") || "[]").length,
      );
      console.log(
        "Deals:",
        JSON.parse(localStorage.getItem("crm-deals") || "[]").length,
      );

      console.log("🔍 Current state data:");
      console.log("Leads:", leads.length);
      console.log("Accounts:", accounts.length);
      console.log("Contacts:", contacts.length);
      console.log("Deals:", deals.length);

      // Show detailed contacts data
      const savedContacts = JSON.parse(
        localStorage.getItem("crm-contacts") || "[]",
      );
      console.log(
        "📋 Saved contacts details:",
        savedContacts.map((c) => `${c.firstName} ${c.lastName} (${c.id})`),
      );
      console.log(
        "📋 Current contacts details:",
        contacts.map((c) => `${c.firstName} ${c.lastName} (${c.id})`),
      );
    };

    // Add a function to reset to initial data
    (window as any).resetCRMData = () => {
      console.log("🔄 Resetting CRM data to initial state...");
      localStorage.removeItem("crm-leads");
      localStorage.removeItem("crm-accounts");
      localStorage.removeItem("crm-contacts");
      localStorage.removeItem("crm-deals");

      setLeads(initialLeads);
      setAccounts(initialAccounts);
      setContacts(initialContacts);
      setDeals(initialDeals);
    };
  }, [leads, accounts, contacts, deals]);

  const value: CRMContextType = {
    leads,
    accounts,
    contacts,
    deals,
    addLead,
    updateLead,
    deleteLead,
    addAccount,
    updateAccount,
    deleteAccount,
    addContact,
    updateContact,
    deleteContact,
    addDeal,
    updateDeal,
    deleteDeal,
  };

  return <CRMContext.Provider value={value}>{children}</CRMContext.Provider>;
}

export function useCRM() {
  const context = useContext(CRMContext);
  if (context === undefined) {
    throw new Error("useCRM must be used within a CRMProvider");
  }
  return context;
}

// Export types for use in components
export type { Lead, Account, Contact, ActiveDeal };
