import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { api } from '@/services/demoApi';
import { Contact } from '@shared/models';

interface ContactDetailProps {
  contactId: string | number;
  onBack: () => void;
}

export default function ContactDetail({ contactId, onBack }: ContactDetailProps) {
  const [formData, setFormData] = useState({
    firstName: '',
    lastName: '',
    title: '',
    associatedAccount: '',
    emailAddress: '',
    deskPhone: '',
    mobilePhone: '',
    city: '',
    state: '',
    country: '',
    timeZone: '',
    source: '',
    owner: '',
    status: ''
  });
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [isNewContact, setIsNewContact] = useState(false);

  useEffect(() => {
    const loadContact = async () => {
      if (typeof contactId === 'string' && contactId !== 'new') {
        setLoading(true);
        try {
          const response = await api.contacts.getById(contactId);
          if (response.success && response.data) {
            const contact = response.data;
            setFormData({
              firstName: contact.firstName || '',
              lastName: contact.lastName || '',
              title: contact.title || '',
              associatedAccount: contact.associatedAccount || '',
              emailAddress: contact.emailAddress || '',
              deskPhone: contact.deskPhone || '',
              mobilePhone: contact.mobilePhone || '',
              city: contact.city || '',
              state: contact.state || '',
              country: contact.country || '',
              timeZone: contact.timeZone || '',
              source: contact.source || '',
              owner: contact.owner || '',
              status: contact.status || ''
            });
          }
        } catch (error) {
          console.error('Error loading contact:', error);
        } finally {
          setLoading(false);
        }
      } else {
        setIsNewContact(true);
      }
    };

    loadContact();
  }, [contactId]);

  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      if (isNewContact || typeof contactId === 'number') {
        // Create new contact
        const response = await api.contacts.create({
          ...formData,
          createdBy: 'current-user',
          updatedBy: 'current-user'
        } as any);
        if (response.success) {
          console.log('Contact created successfully');
          onBack();
        }
      } else {
        // Update existing contact
        const response = await api.contacts.update(contactId as string, formData);
        if (response.success) {
          console.log('Contact updated successfully');
          onBack();
        }
      }
    } catch (error) {
      console.error('Error saving contact:', error);
      alert('Error saving contact. Please try again.');
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="p-6 flex justify-center items-center">
        <div className="text-lg">Loading contact...</div>
      </div>
    );
  }

  return (
    <div className="p-6 bg-gray-50 dark:bg-gray-900 min-h-full">
      <div className="max-w-4xl mx-auto">
        <div className="mb-6 flex items-center justify-between">
          <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            {isNewContact ? 'New Contact' : 'Edit Contact'}
          </h1>
          <Button variant="outline" onClick={onBack}>
            Back to Contacts
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
                  Associated Account
                </label>
                <Input
                  placeholder="Company/Account name"
                  value={formData.associatedAccount}
                  onChange={(e) => handleInputChange('associatedAccount', e.target.value)}
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Source
                </label>
                <Select value={formData.source} onValueChange={(value) => handleInputChange('source', value)}>
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select source" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Data Research">Data Research</SelectItem>
                    <SelectItem value="Referral">Referral</SelectItem>
                    <SelectItem value="Event">Event</SelectItem>
                  </SelectContent>
                </Select>
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
                    <SelectItem value="Suspect">Suspect</SelectItem>
                    <SelectItem value="Prospect">Prospect</SelectItem>
                    <SelectItem value="Active Deal">Active Deal</SelectItem>
                    <SelectItem value="Do Not Call">Do Not Call</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Contact Information</h3>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Email Address
                </label>
                <Input
                  type="email"
                  placeholder="email@example.com"
                  value={formData.emailAddress}
                  onChange={(e) => handleInputChange('emailAddress', e.target.value)}
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Desk Phone
                </label>
                <Input
                  placeholder="+1-555-0123"
                  value={formData.deskPhone}
                  onChange={(e) => handleInputChange('deskPhone', e.target.value)}
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Mobile Phone
                </label>
                <Input
                  placeholder="+1-555-0123"
                  value={formData.mobilePhone}
                  onChange={(e) => handleInputChange('mobilePhone', e.target.value)}
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  City
                </label>
                <Input
                  placeholder="City"
                  value={formData.city}
                  onChange={(e) => handleInputChange('city', e.target.value)}
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  State
                </label>
                <Input
                  placeholder="State"
                  value={formData.state}
                  onChange={(e) => handleInputChange('state', e.target.value)}
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Country
                </label>
                <Input
                  placeholder="Country"
                  value={formData.country}
                  onChange={(e) => handleInputChange('country', e.target.value)}
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Time Zone
                </label>
                <Input
                  placeholder="EST, PST, etc."
                  value={formData.timeZone}
                  onChange={(e) => handleInputChange('timeZone', e.target.value)}
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Owner
                </label>
                <Input
                  placeholder="Contact owner"
                  value={formData.owner}
                  onChange={(e) => handleInputChange('owner', e.target.value)}
                  className="w-full"
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
              disabled={saving || !formData.firstName || !formData.lastName}
            >
              {saving ? 'Saving...' : (isNewContact ? 'Create Contact' : 'Update Contact')}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
