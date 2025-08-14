import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Textarea } from '@/components/ui/textarea';
import { api } from '@/services/demoApi';
import { ActivityLog } from '@shared/models';

interface ActivityDetailProps {
  activityId: string | number;
  onBack: () => void;
}

export default function ActivityDetail({ activityId, onBack }: ActivityDetailProps) {
  const [formData, setFormData] = useState({
    activityType: '',
    associatedContact: '',
    associatedAccount: '',
    dateTime: '',
    followUpSchedule: '',
    summary: '',
    outcomeDisposition: ''
  });
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [isNewActivity, setIsNewActivity] = useState(false);

  useEffect(() => {
    const loadActivity = async () => {
      if (typeof activityId === 'string' && activityId !== 'new') {
        setLoading(true);
        try {
          const response = await api.activities.getById(activityId);
          if (response.success && response.data) {
            const activity = response.data;
            setFormData({
              activityType: activity.activityType || '',
              associatedContact: activity.associatedContact || '',
              associatedAccount: activity.associatedAccount || '',
              dateTime: activity.dateTime ? new Date(activity.dateTime).toISOString().slice(0, 16) : '',
              followUpSchedule: activity.followUpSchedule || '',
              summary: activity.summary || '',
              outcomeDisposition: activity.outcomeDisposition || ''
            });
          }
        } catch (error) {
          console.error('Error loading activity:', error);
        } finally {
          setLoading(false);
        }
      } else {
        setIsNewActivity(true);
        setFormData({
          ...formData,
          dateTime: new Date().toISOString().slice(0, 16)
        });
      }
    };

    loadActivity();
  }, [activityId]);

  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      const activityData = {
        ...formData,
        dateTime: new Date(formData.dateTime),
        createdBy: 'current-user',
        updatedBy: 'current-user'
      };

      if (isNewActivity || typeof activityId === 'number') {
        const response = await api.activities.create(activityData as any);
        if (response.success) {
          console.log('Activity created successfully');
          onBack();
        }
      } else {
        const response = await api.activities.update(activityId as string, activityData);
        if (response.success) {
          console.log('Activity updated successfully');
          onBack();
        }
      }
    } catch (error) {
      console.error('Error saving activity:', error);
      alert('Error saving activity. Please try again.');
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="p-6 flex justify-center items-center">
        <div className="text-lg">Loading activity...</div>
      </div>
    );
  }

  return (
    <div className="p-6 bg-gray-50 dark:bg-gray-900 min-h-full">
      <div className="max-w-4xl mx-auto">
        <div className="mb-6 flex items-center justify-between">
          <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            {isNewActivity ? 'New Activity' : 'Edit Activity'}
          </h1>
          <Button variant="outline" onClick={onBack}>
            Back to Activities
          </Button>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Activity Information</h3>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Activity Type *
                </label>
                <Select value={formData.activityType} onValueChange={(value) => handleInputChange('activityType', value)}>
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select activity type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Call">Call</SelectItem>
                    <SelectItem value="Email">Email</SelectItem>
                    <SelectItem value="LinkedIn Msg">LinkedIn Msg</SelectItem>
                    <SelectItem value="SMS">SMS</SelectItem>
                    <SelectItem value="Other">Other</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Date & Time *
                </label>
                <Input
                  type="datetime-local"
                  value={formData.dateTime}
                  onChange={(e) => handleInputChange('dateTime', e.target.value)}
                  className="w-full"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Associated Contact
                </label>
                <Input
                  placeholder="Contact name"
                  value={formData.associatedContact}
                  onChange={(e) => handleInputChange('associatedContact', e.target.value)}
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Associated Account
                </label>
                <Input
                  placeholder="Account name"
                  value={formData.associatedAccount}
                  onChange={(e) => handleInputChange('associatedAccount', e.target.value)}
                  className="w-full"
                />
              </div>
            </div>

            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Activity Details</h3>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Outcome/Disposition
                </label>
                <Select value={formData.outcomeDisposition} onValueChange={(value) => handleInputChange('outcomeDisposition', value)}>
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select outcome" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Voicemail">Voicemail</SelectItem>
                    <SelectItem value="RNR">RNR</SelectItem>
                    <SelectItem value="Meeting Fixed">Meeting Fixed</SelectItem>
                    <SelectItem value="Meeting Completed">Meeting Completed</SelectItem>
                    <SelectItem value="Meeting Rescheduled">Meeting Rescheduled</SelectItem>
                    <SelectItem value="Not Interested">Not Interested</SelectItem>
                    <SelectItem value="Do not Call">Do not Call</SelectItem>
                    <SelectItem value="Callback requested">Callback requested</SelectItem>
                    <SelectItem value="Email sent">Email sent</SelectItem>
                    <SelectItem value="Email Received">Email Received</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Follow-up Schedule
                </label>
                <Input
                  placeholder="Follow-up details"
                  value={formData.followUpSchedule}
                  onChange={(e) => handleInputChange('followUpSchedule', e.target.value)}
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Summary
                </label>
                <Textarea
                  placeholder="Activity summary and notes"
                  value={formData.summary}
                  onChange={(e) => handleInputChange('summary', e.target.value)}
                  className="w-full"
                  rows={6}
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
              disabled={saving || !formData.activityType || !formData.dateTime}
            >
              {saving ? 'Saving...' : (isNewActivity ? 'Create Activity' : 'Update Activity')}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
