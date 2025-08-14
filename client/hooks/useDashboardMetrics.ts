import { useState, useEffect } from "react";
import { useAuth } from "@/components/RealAuthProvider";
import { api } from "@/services/demoApi";

interface DashboardMetrics {
  totalRevenue: string;
  openActiveDeals: string;
  newLeads: string;
  closedDeals: string;
  loading: boolean;
}

export function useDashboardMetrics(
  userRole?: "admin" | "user" | null,
): DashboardMetrics {
  const [metrics, setMetrics] = useState<DashboardMetrics>({
    totalRevenue: "$",
    openActiveDeals: "",
    newLeads: "",
    closedDeals: "",
    loading: true,
  });

  const { user } = useAuth();

  useEffect(() => {
    const loadMetrics = async () => {
      try {
        const [dealsResponse, leadsResponse] = await Promise.all([
          api.deals.getAll(),
          api.leads.getAll(),
        ]);

        let totalRevenue = 0;
        let openActiveDeals = 0;
        let closedDeals = 0;

        if (dealsResponse.success) {
          let dealsData = dealsResponse.data;

          // Filter deals based on user role
          if (userRole === "user" && user?.primaryEmail) {
            dealsData = dealsData.filter(
              (deal) =>
                deal.dealOwner === user.primaryEmail ||
                deal.createdBy === user.primaryEmail,
            );
          }
          // Admin sees all deals

          dealsData.forEach((deal) => {
            // Count open active deals (all stages except closed won/lost)
            if (
              deal.stage &&
              ![
                "Order Won",
                "Order Lost",
                "Closed Won",
                "Closed Lost",
              ].includes(deal.stage)
            ) {
              openActiveDeals++;
            }

            // Count closed deals and calculate revenue
            if (deal.stage === "Order Won" || deal.stage === "Closed Won") {
              closedDeals++;
              // Extract numeric value from deal value
              const valueStr = deal.dealValue?.toString() || "0";
              const value = valueStr.replace(/[^0-9.-]+/g, "") || "0";
              const numValue = parseFloat(value);
              if (!isNaN(numValue)) {
                totalRevenue += numValue;
              }
            }
          });
        }

        let newLeads = 0;
        if (leadsResponse.success) {
          let leadsData = leadsResponse.data;

          // Filter leads based on user role
          if (userRole === "user" && user?.primaryEmail) {
            leadsData = leadsData.filter(
              (lead) =>
                lead.leadOwner === user.primaryEmail ||
                lead.createdBy === user.primaryEmail,
            );
          }
          // Admin sees all leads

          newLeads = leadsData.length;
        }

        setMetrics({
          totalRevenue:
            totalRevenue > 0 ? `$${totalRevenue.toLocaleString()}` : "$",
          openActiveDeals: openActiveDeals.toString(),
          newLeads: newLeads.toString(),
          closedDeals: closedDeals.toString(),
          loading: false,
        });
      } catch (error) {
        console.error("Error loading dashboard metrics:", error);
        setMetrics((prev) => ({ ...prev, loading: false }));
      }
    };

    loadMetrics();
  }, [userRole, user]);

  return metrics;
}
