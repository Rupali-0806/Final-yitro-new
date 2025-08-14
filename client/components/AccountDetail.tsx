import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { api } from '@/services/demoApi';
import { Account } from '@shared/models';

interface AccountDetailProps {
  accountId: string | number;
  onBack: () => void;
}

export default function AccountDetail({ accountId, onBack }: AccountDetailProps) {
  const [formData, setFormData] = useState({
    accountName: '',
    accountRating: '',
    accountOwner: '',
    status: '',
    industry: '',
    revenue: '',
    numberOfEmployees: '',
    addressLine1: '',
    addressLine2: '',
    city: '',
    state: '',
    country: '',
    zipPostCode: '',
    timeZone: '',
    boardNumber: '',
    website: '',
    geo: ''
  });
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [isNewAccount, setIsNewAccount] = useState(false);

  useEffect(() => {
    const loadAccount = async () => {
      if (typeof accountId === 'string' && accountId !== 'new') {
        setLoading(true);
        try {
          const response = await api.accounts.getById(accountId);
          if (response.success && response.data) {
            const account = response.data;
            setFormData({
              accountName: account.accountName || '',
              accountRating: account.accountRating || '',
              accountOwner: account.accountOwner || '',
              status: account.status || '',
              industry: account.industry || '',
              revenue: account.revenue || '',
              numberOfEmployees: account.numberOfEmployees || '',
              addressLine1: account.addressLine1 || '',
              addressLine2: account.addressLine2 || '',
              city: account.city || '',
              state: account.state || '',
              country: account.country || '',
              zipPostCode: account.zipPostCode || '',
              timeZone: account.timeZone || '',
              boardNumber: account.boardNumber || '',
              website: account.website || '',
              geo: account.geo || ''
            });
          }
        } catch (error) {
          console.error('Error loading account:', error);
        } finally {
          setLoading(false);
        }
      } else {
        setIsNewAccount(true);
      }
    };

    loadAccount();
  }, [accountId]);

  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      if (isNewAccount || typeof accountId === 'number') {
        // Create new account
        const response = await api.accounts.create({
          ...formData,
          createdBy: 'current-user',
          updatedBy: 'current-user'
        } as any);
        if (response.success) {
          console.log('Account created successfully');
          onBack();
        }
      } else {
        // Update existing account
        const response = await api.accounts.update(accountId as string, formData);
        if (response.success) {
          console.log('Account updated successfully');
          onBack();
        }
      }
    } catch (error) {
      console.error('Error saving account:', error);
      alert('Error saving account. Please try again.');
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="p-6 flex justify-center items-center">
        <div className="text-lg">Loading account...</div>
      </div>
    );
  }

  return (
    <div className="p-6 bg-gray-50 dark:bg-gray-900 min-h-full">
      <div className="max-w-4xl mx-auto">
        <div className="mb-6 flex items-center justify-between">
          <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            {isNewAccount ? 'New Account' : 'Edit Account'}
          </h1>
          <Button variant="outline" onClick={onBack}>
            Back to Accounts
          </Button>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Basic Information</h3>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Account Name *
                </label>
                <Input
                  placeholder="Enter account name"
                  value={formData.accountName}
                  onChange={(e) => handleInputChange('accountName', e.target.value)}
                  className="w-full"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Account Rating
                </label>
                <Select value={formData.accountRating} onValueChange={(value) => handleInputChange('accountRating', value)}>
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select rating" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Platinum (Must Have)">Platinum (Must Have)</SelectItem>
                    <SelectItem value="Gold (High Priority)">Gold (High Priority)</SelectItem>
                    <SelectItem value="Silver (Medium Priority)">Silver (Medium Priority)</SelectItem>
                    <SelectItem value="Bronze (Low Priority)">Bronze (Low Priority)</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Account Owner
                </label>
                <Input
                  placeholder="Account owner name"
                  value={formData.accountOwner}
                  onChange={(e) => handleInputChange('accountOwner', e.target.value)}
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
                    <SelectItem value="Suspect">Suspect</SelectItem>
                    <SelectItem value="Prospect">Prospect</SelectItem>
                    <SelectItem value="Active Deal">Active Deal</SelectItem>
                    <SelectItem value="Do Not Call">Do Not Call</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Industry
                </label>
                <Input
                  placeholder="Industry sector"
                  value={formData.industry}
                  onChange={(e) => handleInputChange('industry', e.target.value)}
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Revenue
                </label>
                <Select value={formData.revenue} onValueChange={(value) => handleInputChange('revenue', value)}>
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select revenue range" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Under $1M">Under $1M</SelectItem>
                    <SelectItem value="$1M - $10M">$1M - $10M</SelectItem>
                    <SelectItem value="$10M - $50M">$10M - $50M</SelectItem>
                    <SelectItem value="$50M+">$50M+</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Number of Employees
                </label>
                <Select value={formData.numberOfEmployees} onValueChange={(value) => handleInputChange('numberOfEmployees', value)}>
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select employee range" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="1-10">1-10</SelectItem>
                    <SelectItem value="11-50">11-50</SelectItem>
                    <SelectItem value="51-200">51-200</SelectItem>
                    <SelectItem value="201-1000">201-1000</SelectItem>
                    <SelectItem value="1000+">1000+</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Geography
                </label>
                <Select value={formData.geo} onValueChange={(value) => handleInputChange('geo', value)}>
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
            </div>

            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Address Information</h3>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Address Line 1
                </label>
                <Input
                  placeholder="Street address"
                  value={formData.addressLine1}
                  onChange={(e) => handleInputChange('addressLine1', e.target.value)}
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Address Line 2
                </label>
                <Input
                  placeholder="Apt, suite, etc."
                  value={formData.addressLine2}
                  onChange={(e) => handleInputChange('addressLine2', e.target.value)}
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
                  placeholder="State/Province"
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
                  ZIP/Postal Code
                </label>
                <Input
                  placeholder="ZIP/Postal code"
                  value={formData.zipPostCode}
                  onChange={(e) => handleInputChange('zipPostCode', e.target.value)}
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
                  Board Number
                </label>
                <Input
                  placeholder="Board number"
                  value={formData.boardNumber}
                  onChange={(e) => handleInputChange('boardNumber', e.target.value)}
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Website
                </label>
                <Input
                  placeholder="https://example.com"
                  value={formData.website}
                  onChange={(e) => handleInputChange('website', e.target.value)}
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
              disabled={saving || !formData.accountName}
            >
              {saving ? 'Saving...' : (isNewAccount ? 'Create Account' : 'Update Account')}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
