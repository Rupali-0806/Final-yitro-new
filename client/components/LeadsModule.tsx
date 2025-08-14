import { useState, useEffect } from 'react';
import ListView from './ListView';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { api } from '@/services/demoApi';
import { Lead } from '@shared/models';

interface LeadsModuleProps {
  view: 'list' | 'detail';
  selectedLead?: Lead;
  onViewChange: (view: 'list' | 'detail', lead?: Lead) => void;
  onBack: () => void;
}

export default function LeadsModule({ view, selectedLead, onViewChange, onBack }: LeadsModuleProps) {
  const [leads, setLeads] = useState<Lead[]>([]);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [isNewLead, setIsNewLead] = useState(false);
  const [formData, setFormData] = useState({
    firstName: '',
    lastName: '',
    company: '',
    title: '',
    phone: '',
    email: '',
    leadSource: '',
    status: '',
    rating: '',
    owner: ''
  });

  useEffect(() => {
    if (view === 'list') {
      loadLeads();
    } else if (view === 'detail') {
      if (selectedLead) {
        setFormData({
          firstName: selectedLead.firstName || '',
          lastName: selectedLead.lastName || '',
          company: selectedLead.company || '',
          title: selectedLead.title || '',
          phone: selectedLead.phone || '',
          email: selectedLead.email || '',
          leadSource: selectedLead.leadSource || '',
          status: selectedLead.status || '',
          rating: selectedLead.rating || '',
          owner: selectedLead.owner || ''
        });
        setIsNewLead(false);
      } else {
        setFormData({
          firstName: '',
          lastName: '',
          company: '',
          title: '',
          phone: '',
          email: '',
          leadSource: '',
          status: '',
          rating: '',
          owner: ''
        });
        setIsNewLead(true);
      }
    }
  }, [view, selectedLead]);

  const loadLeads = async () => {
    setLoading(true);
    try {
      const response = await api.leads.getAll();
      if (response.success) {
        setLeads(response.data);
      }
    } catch (error) {
      console.error('Error loading leads:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      if (isNewLead) {
        const response = await api.leads.create({
          ...formData,
          createdBy: 'current-user',
          updatedBy: 'current-user'
        } as any);
        if (response.success) {
          console.log('Lead created successfully');
          onViewChange('list');
        }
      } else if (selectedLead) {
        const response = await api.leads.update(selectedLead.id, formData);
        if (response.success) {
          console.log('Lead updated successfully');
          onViewChange('list');
        }
      }
    } catch (error) {
      console.error('Error saving lead:', error);
      alert('Error saving lead. Please try again.');
    } finally {
      setSaving(false);
    }
  };

  const handleEdit = (lead: Lead) => {
    onViewChange('detail', lead);
  };

  const handleDelete = async (lead: Lead) => {
    if (confirm(`Are you sure you want to delete ${lead.firstName} ${lead.lastName}?`)) {
      try {
        await api.leads.delete(lead.id);
        console.log('Lead deleted successfully');
        loadLeads();
      } catch (error) {
        console.error('Error deleting lead:', error);
      }
    }
  };

  if (view === 'list') {
    return (
      <ListView
        title="Leads"
        columns={[
          { key: 'firstName', label: 'First Name', sortable: true, width: '15%' },
          { key: 'lastName', label: 'Last Name', sortable: true, width: '15%' },
          { key: 'company', label: 'Company', sortable: true, width: '20%' },
          { key: 'title', label: 'Title', sortable: true, width: '15%' },
          { key: 'email', label: 'Email', sortable: false, width: '20%' },
          { key: 'status', label: 'Status', sortable: true, width: '10%' },
          { key: 'rating', label: 'Rating', sortable: true, width: '5%' }
        ]}
        records={leads}
        onRecordClick={(record) => onViewChange('detail', record)}
        onNewRecord={() => onViewChange('detail')}
        onEditRecord={handleEdit}
        onDeleteRecord={handleDelete}
        loading={loading}
      />
    );
  }

  return (
    <div className="p-6 bg-gray-50 dark:bg-gray-900 min-h-full">
      <div className="max-w-4xl mx-auto">
        <div className="mb-6 flex items-center justify-between">
          <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            {isNewLead ? 'New Lead' : 'Edit Lead'}
          </h1>
          <Button variant="outline" onClick={() => onViewChange('list')}>
            Back to Leads
          </Button>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Basic Information</h3>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  First Name *
                </label>
                <Input
                  placeholder="Enter first name"
                  value={formData.firstName}
                  onChange={(e) => handleInputChange('firstName', e.target.value)}
                  className="w-full"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Last Name *
                </label>
                <Input
                  placeholder="Enter last name"
                  value={formData.lastName}
                  onChange={(e) => handleInputChange('lastName', e.target.value)}
                  className="w-full"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Company *
                </label>
                <Input
                  placeholder="Company name"
                  value={formData.company}
                  onChange={(e) => handleInputChange('company', e.target.value)}
                  className="w-full"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Title
                </label>
                <Input
                  placeholder="Job title"
                  value={formData.title}
                  onChange={(e) => handleInputChange('title', e.target.value)}
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Lead Source
                </label>
                <Select value={formData.leadSource} onValueChange={(value) => handleInputChange('leadSource', value)}>
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select lead source" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Website">Website</SelectItem>
                    <SelectItem value="Referral">Referral</SelectItem>
                    <SelectItem value="Trade Show">Trade Show</SelectItem>
                    <SelectItem value="Cold Call">Cold Call</SelectItem>
                    <SelectItem value="Email">Email</SelectItem>
                    <SelectItem value="Partner">Partner</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Contact Information</h3>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Phone
                </label>
                <Input
                  placeholder="+1-555-0123"
                  value={formData.phone}
                  onChange={(e) => handleInputChange('phone', e.target.value)}
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Email
                </label>
                <Input
                  type="email"
                  placeholder="email@example.com"
                  value={formData.email}
                  onChange={(e) => handleInputChange('email', e.target.value)}
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Status
                </label>
                <Select value={formData.status} onValueChange={(value) => handleInputChange('status', value)}>
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select status" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="New">New</SelectItem>
                    <SelectItem value="Working">Working</SelectItem>
                    <SelectItem value="Qualified">Qualified</SelectItem>
                    <SelectItem value="Unqualified">Unqualified</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Rating
                </label>
                <Select value={formData.rating} onValueChange={(value) => handleInputChange('rating', value)}>
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select rating" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Hot">Hot</SelectItem>
                    <SelectItem value="Warm">Warm</SelectItem>
                    <SelectItem value="Cold">Cold</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Owner
                </label>
                <Input
                  placeholder="Lead owner"
                  value={formData.owner}
                  onChange={(e) => handleInputChange('owner', e.target.value)}
                  className="w-full"
                />
              </div>
            </div>
          </div>

          <div className="flex justify-end space-x-4 pt-6 border-t">
            <Button variant="outline" onClick={() => onViewChange('list')}>
              Cancel
            </Button>
            <Button 
              onClick={handleSave} 
              disabled={saving || !formData.firstName || !formData.lastName || !formData.company}
            >
              {saving ? 'Saving...' : (isNewLead ? 'Create Lead' : 'Update Lead')}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
