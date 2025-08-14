import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { api } from "@/services/demoApi";
import { ActiveDeal } from "@shared/models";

interface DealDetailProps {
  dealId: string | number;
  onBack: () => void;
}

export default function DealDetail({ dealId, onBack }: DealDetailProps) {
  const [formData, setFormData] = useState({
    dealOwner: "",
    dealName: "",
    businessLine: "",
    associatedAccount: "",
    associatedContact: "",
    closingDate: "",
    probability: "",
    dealValue: "",
    approvedBy: "",
    description: "",
    nextStep: "",
    geo: "",
    entity: "",
    stage: "",
  });
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [isNewDeal, setIsNewDeal] = useState(false);

  useEffect(() => {
    const loadDeal = async () => {
      if (typeof dealId === "string" && dealId !== "new") {
        setLoading(true);
        try {
          const response = await api.deals.getById(dealId);
          if (response.success && response.data) {
            const deal = response.data;
            setFormData({
              dealOwner: deal.dealOwner || "",
              dealName: deal.dealName || "",
              businessLine: deal.businessLine || "",
              associatedAccount: deal.associatedAccount || "",
              associatedContact: deal.associatedContact || "",
              closingDate: deal.closingDate
                ? new Date(deal.closingDate).toISOString().split("T")[0]
                : "",
              probability: deal.probability || "",
              dealValue: deal.dealValue || "",
              approvedBy: deal.approvedBy || "",
              description: deal.description || "",
              nextStep: deal.nextStep || "",
              geo: deal.geo || "",
              entity: deal.entity || "",
              stage: deal.stage || "",
            });
          }
        } catch (error) {
          console.error("Error loading deal:", error);
        } finally {
          setLoading(false);
        }
      } else {
        setIsNewDeal(true);
      }
    };

    loadDeal();
  }, [dealId]);

  const handleInputChange = (field: string, value: string) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      const dealData = {
        ...formData,
        closingDate: formData.closingDate
          ? new Date(formData.closingDate)
          : undefined,
        createdBy: "current-user",
        updatedBy: "current-user",
      };

      if (isNewDeal || typeof dealId === "number") {
        // Create new deal
        const response = await api.deals.create(dealData as any);
        if (response.success) {
          console.log("Deal created successfully");
          onBack();
        }
      } else {
        // Update existing deal
        const response = await api.deals.update(dealId as string, dealData);
        if (response.success) {
          console.log("Deal updated successfully");
          onBack();
        }
      }
    } catch (error) {
      console.error("Error saving deal:", error);
      alert("Error saving deal. Please try again.");
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="p-6 flex justify-center items-center">
        <div className="text-lg">Loading deal...</div>
      </div>
    );
  }

  return (
    <div className="p-6 bg-gray-50 dark:bg-gray-900 min-h-full">
      <div className="max-w-4xl mx-auto">
        <div className="mb-6 flex items-center justify-between">
          <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            {isNewDeal ? "New Opportunity" : "Edit Opportunity"}
          </h1>
          <Button variant="outline" onClick={onBack}>
            Back to Active Deals
          </Button>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                Deal Information
              </h3>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Deal Name *
                </label>
                <Input
                  placeholder="Enter deal name"
                  value={formData.dealName}
                  onChange={(e) =>
                    handleInputChange("dealName", e.target.value)
                  }
                  className="w-full"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Deal Owner
                </label>
                <Input
                  placeholder="Deal owner name"
                  value={formData.dealOwner}
                  onChange={(e) =>
                    handleInputChange("dealOwner", e.target.value)
                  }
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Business Line
                </label>
                <Select
                  value={formData.businessLine}
                  onValueChange={(value) =>
                    handleInputChange("businessLine", value)
                  }
                >
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select business line" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Human Capital">Human Capital</SelectItem>
                    <SelectItem value="Managed Services">
                      Managed Services
                    </SelectItem>
                    <SelectItem value="GCC">GCC</SelectItem>
                    <SelectItem value="Automation">Automation</SelectItem>
                    <SelectItem value="Support">Support</SelectItem>
                    <SelectItem value="Product">Product</SelectItem>
                    <SelectItem value="Solution">Solution</SelectItem>
                    <SelectItem value="RCM">RCM</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Associated Account
                </label>
                <Input
                  placeholder="Account name"
                  value={formData.associatedAccount}
                  onChange={(e) =>
                    handleInputChange("associatedAccount", e.target.value)
                  }
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Associated Contact
                </label>
                <Input
                  placeholder="Contact name"
                  value={formData.associatedContact}
                  onChange={(e) =>
                    handleInputChange("associatedContact", e.target.value)
                  }
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Closing Date
                </label>
                <Input
                  type="date"
                  value={formData.closingDate}
                  onChange={(e) =>
                    handleInputChange("closingDate", e.target.value)
                  }
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Probability
                </label>
                <Select
                  value={formData.probability}
                  onValueChange={(value) =>
                    handleInputChange("probability", value)
                  }
                >
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select probability" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="30-100%">30-100%</SelectItem>
                    <SelectItem value="10%">10%</SelectItem>
                    <SelectItem value="25%">25%</SelectItem>
                    <SelectItem value="50%">50%</SelectItem>
                    <SelectItem value="75%">75%</SelectItem>
                    <SelectItem value="90%">90%</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Deal Value
                </label>
                <Input
                  placeholder="$100,000"
                  value={formData.dealValue}
                  onChange={(e) =>
                    handleInputChange("dealValue", e.target.value)
                  }
                  className="w-full"
                />
              </div>
            </div>

            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                Deal Details
              </h3>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Stage
                </label>
                <Select
                  value={formData.stage}
                  onValueChange={(value) => handleInputChange("stage", value)}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select stage" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Opportunity Identified">
                      Opportunity Identified
                    </SelectItem>
                    <SelectItem value="Proposal Submitted">
                      Proposal Submitted
                    </SelectItem>
                    <SelectItem value="Negotiating">Negotiating</SelectItem>
                    <SelectItem value="Closing">Closing</SelectItem>
                    <SelectItem value="Order Won">Order Won</SelectItem>
                    <SelectItem value="Order Lost">Order Lost</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Geography
                </label>
                <Select
                  value={formData.geo}
                  onValueChange={(value) => handleInputChange("geo", value)}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select geography" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Americas">Americas</SelectItem>
                    <SelectItem value="India">India</SelectItem>
                    <SelectItem value="Philippines">Philippines</SelectItem>
                    <SelectItem value="EMEA">EMEA</SelectItem>
                    <SelectItem value="ANZ">ANZ</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Entity
                </label>
                <Select
                  value={formData.entity}
                  onValueChange={(value) => handleInputChange("entity", value)}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select entity" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Yitro Global">Yitro Global</SelectItem>
                    <SelectItem value="Yitro Tech">Yitro Tech</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Approved By
                </label>
                <Input
                  placeholder="Approval manager"
                  value={formData.approvedBy}
                  onChange={(e) =>
                    handleInputChange("approvedBy", e.target.value)
                  }
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Description
                </label>
                <Textarea
                  placeholder="Deal description"
                  value={formData.description}
                  onChange={(e) =>
                    handleInputChange("description", e.target.value)
                  }
                  className="w-full"
                  rows={4}
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Next Step
                </label>
                <Textarea
                  placeholder="Next steps to advance the deal"
                  value={formData.nextStep}
                  onChange={(e) =>
                    handleInputChange("nextStep", e.target.value)
                  }
                  className="w-full"
                  rows={3}
                />
              </div>
            </div>
          </div>

          <div className="flex justify-end space-x-4 pt-6 border-t">
            <Button variant="outline" onClick={onBack}>
              Cancel
            </Button>
            <Button
              onClick={handleSave}
              disabled={saving || !formData.dealName}
            >
              {saving
                ? "Saving..."
                : isNewDeal
                  ? "Create Opportunity"
                  : "Update Opportunity"}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
