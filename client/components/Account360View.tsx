import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Avatar, AvatarFallback, AvatarInitials } from '@/components/ui/avatar';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  Building2,
  Users,
  Activity,
  DollarSign,
  Calendar,
  Phone,
  Mail,
  MapPin,
  Globe,
  ArrowLeft,
  TrendingUp,
  Target,
  ChevronRight,
  Plus,
  Filter,
  Edit3
} from 'lucide-react';
import { api } from '@/services/demoApi';
import { Account, Contact, ActiveDeal, ActivityLog } from '@shared/models';
import { format } from 'date-fns';

interface Account360ViewProps {
  accountId: string;
  onBack: () => void;
  onEdit: () => void;
}

interface RelatedData {
  contacts: Contact[];
  deals: ActiveDeal[];
  activities: ActivityLog[];
}

export default function Account360View({ accountId, onBack, onEdit }: Account360ViewProps) {
  const [account, setAccount] = useState<Account | null>(null);
  const [relatedData, setRelatedData] = useState<RelatedData>({
    contacts: [],
    deals: [],
    activities: []
  });
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('overview');

  useEffect(() => {
    loadAccountData();
  }, [accountId]);

  const loadAccountData = async () => {
    setLoading(true);
    try {
      // Load account details
      const accountResponse = await api.accounts.getById(accountId);
      if (accountResponse.success && accountResponse.data) {
        setAccount(accountResponse.data);
      }

      // Load related data in parallel
      const [contactsResponse, dealsResponse, activitiesResponse] = await Promise.all([
        api.contacts.getAll(),
        api.deals.getAll(),
        api.activities.getAll()
      ]);

      // Filter related data by account ID
      const relatedContacts = contactsResponse.success ? 
        contactsResponse.data.filter(contact => contact.associatedAccount === accountId) : [];
      
      const relatedDeals = dealsResponse.success ? 
        dealsResponse.data.filter(deal => deal.associatedAccount === accountId) : [];
      
      const relatedActivities = activitiesResponse.success ? 
        activitiesResponse.data.filter(activity => activity.associatedAccount === accountId) : [];

      setRelatedData({
        contacts: relatedContacts,
        deals: relatedDeals,
        activities: relatedActivities.sort((a, b) => new Date(b.dateTime).getTime() - new Date(a.dateTime).getTime())
      });
    } catch (error) {
      console.error('Error loading account data:', error);
    } finally {
      setLoading(false);
    }
  };

  const getAccountHealthScore = () => {
    let score = 50; // Base score
    
    if (account?.accountRating === 'Platinum (Must Have)') score += 25;
    else if (account?.accountRating === 'Gold (High Priority)') score += 15;
    else if (account?.accountRating === 'Silver (Medium Priority)') score += 5;
    
    if (account?.status === 'Active Deal') score += 20;
    else if (account?.status === 'Prospect') score += 10;
    
    if (relatedData.deals.length > 0) score += 15;
    if (relatedData.activities.length > 2) score += 10;
    
    return Math.min(score, 100);
  };

  const getTotalDealValue = () => {
    return relatedData.deals.reduce((total, deal) => {
      const value = parseFloat(deal.dealValue?.replace(/[^0-9.-]+/g, '') || '0');
      return total + value;
    }, 0);
  };

  const getRecentActivities = () => {
    return relatedData.activities.slice(0, 5);
  };

  if (loading) {
    return (
      <div className="p-6 flex justify-center items-center min-h-[400px]">
        <div className="text-lg">Loading account 360 view...</div>
      </div>
    );
  }

  if (!account) {
    return (
      <div className="p-6 flex justify-center items-center min-h-[400px]">
        <div className="text-lg text-gray-500">Account not found</div>
      </div>
    );
  }

  const healthScore = getAccountHealthScore();
  const totalDealValue = getTotalDealValue();

  return (
    <div className="p-6 bg-gray-50 dark:bg-gray-900 min-h-full">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-6 flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <Button variant="outline" size="sm" onClick={onBack}>
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back
            </Button>
            <div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">
                {account.accountName}
              </h1>
              <p className="text-gray-600 dark:text-gray-400 mt-1">
                Account 360° View
              </p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <Badge 
              variant={account.status === 'Active Deal' ? 'default' : 'secondary'}
              className="text-sm"
            >
              {account.status}
            </Badge>
            <Badge 
              variant={account.accountRating?.includes('Platinum') ? 'default' : 'outline'}
              className="text-sm"
            >
              {account.accountRating}
            </Badge>
            <Button onClick={onEdit} size="sm">
              <Edit3 className="w-4 h-4 mr-2" />
              Edit Account
            </Button>
          </div>
        </div>

        {/* Key Metrics Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-blue-100 dark:bg-blue-900 rounded-lg">
                  <TrendingUp className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                </div>
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Health Score</p>
                  <p className="text-2xl font-bold">{healthScore}%</p>
                </div>
              </div>
              <Progress value={healthScore} className="mt-3" />
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-green-100 dark:bg-green-900 rounded-lg">
                  <DollarSign className="w-5 h-5 text-green-600 dark:text-green-400" />
                </div>
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Deal Value</p>
                  <p className="text-2xl font-bold">${totalDealValue.toLocaleString()}</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-purple-100 dark:bg-purple-900 rounded-lg">
                  <Users className="w-5 h-5 text-purple-600 dark:text-purple-400" />
                </div>
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Contacts</p>
                  <p className="text-2xl font-bold">{relatedData.contacts.length}</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-orange-100 dark:bg-orange-900 rounded-lg">
                  <Activity className="w-5 h-5 text-orange-600 dark:text-orange-400" />
                </div>
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Activities</p>
                  <p className="text-2xl font-bold">{relatedData.activities.length}</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Tabbed Content */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="grid w-full grid-cols-5">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="contacts">Contacts ({relatedData.contacts.length})</TabsTrigger>
            <TabsTrigger value="deals">Deals ({relatedData.deals.length})</TabsTrigger>
            <TabsTrigger value="activities">Activities ({relatedData.activities.length})</TabsTrigger>
            <TabsTrigger value="insights">Insights</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Account Details */}
              <Card className="lg:col-span-2">
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <Building2 className="w-5 h-5" />
                    <span>Account Information</span>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-sm text-gray-600 dark:text-gray-400">Owner</p>
                      <p className="font-medium">{account.accountOwner || 'Not assigned'}</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-600 dark:text-gray-400">Industry</p>
                      <p className="font-medium">{account.industry || 'Not specified'}</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-600 dark:text-gray-400">Revenue</p>
                      <p className="font-medium">{account.revenue || 'Not specified'}</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-600 dark:text-gray-400">Employees</p>
                      <p className="font-medium">{account.numberOfEmployees || 'Not specified'}</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-600 dark:text-gray-400">Geography</p>
                      <p className="font-medium">{account.geo || 'Not specified'}</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-600 dark:text-gray-400">Website</p>
                      {account.website ? (
                        <a 
                          href={account.website} 
                          target="_blank" 
                          rel="noopener noreferrer"
                          className="font-medium text-blue-600 hover:underline flex items-center"
                        >
                          <Globe className="w-4 h-4 mr-1" />
                          {account.website}
                        </a>
                      ) : (
                        <p className="font-medium">Not provided</p>
                      )}
                    </div>
                  </div>
                  
                  {(account.addressLine1 || account.city) && (
                    <>
                      <Separator />
                      <div>
                        <p className="text-sm text-gray-600 dark:text-gray-400 mb-2 flex items-center">
                          <MapPin className="w-4 h-4 mr-1" />
                          Address
                        </p>
                        <div className="text-sm">
                          {account.addressLine1 && <p>{account.addressLine1}</p>}
                          {account.addressLine2 && <p>{account.addressLine2}</p>}
                          <p>
                            {[account.city, account.state, account.zipPostCode].filter(Boolean).join(', ')}
                          </p>
                          {account.country && <p>{account.country}</p>}
                        </div>
                      </div>
                    </>
                  )}
                </CardContent>
              </Card>

              {/* Recent Activities */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center justify-between">
                    <span className="flex items-center space-x-2">
                      <Activity className="w-5 h-5" />
                      <span>Recent Activities</span>
                    </span>
                    <Button variant="outline" size="sm">
                      View All
                    </Button>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-[300px]">
                    <div className="space-y-3">
                      {getRecentActivities().map((activity) => (
                        <div key={activity.id} className="flex items-start space-x-3 p-3 rounded-lg bg-gray-50 dark:bg-gray-800">
                          <div className="p-1 bg-blue-100 dark:bg-blue-900 rounded">
                            {activity.activityType === 'Call' && <Phone className="w-3 h-3 text-blue-600" />}
                            {activity.activityType === 'Email' && <Mail className="w-3 h-3 text-blue-600" />}
                            {activity.activityType === 'LinkedIn Msg' && <Activity className="w-3 h-3 text-blue-600" />}
                            {!['Call', 'Email', 'LinkedIn Msg'].includes(activity.activityType) && <Activity className="w-3 h-3 text-blue-600" />}
                          </div>
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-medium">{activity.activityType}</p>
                            <p className="text-xs text-gray-600 dark:text-gray-400 truncate">
                              {activity.summary || 'No summary provided'}
                            </p>
                            <p className="text-xs text-gray-500 mt-1">
                              {format(new Date(activity.dateTime), 'MMM d, h:mm a')}
                            </p>
                          </div>
                        </div>
                      ))}
                      {getRecentActivities().length === 0 && (
                        <p className="text-sm text-gray-500 text-center py-4">
                          No recent activities
                        </p>
                      )}
                    </div>
                  </ScrollArea>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="contacts" className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold">Related Contacts</h3>
              <div className="flex items-center space-x-2">
                <Button variant="outline" size="sm">
                  <Filter className="w-4 h-4 mr-2" />
                  Filter
                </Button>
                <Button size="sm">
                  <Plus className="w-4 h-4 mr-2" />
                  Add Contact
                </Button>
              </div>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {relatedData.contacts.map((contact) => (
                <Card key={contact.id}>
                  <CardContent className="p-4">
                    <div className="flex items-start space-x-3">
                      <Avatar>
                        <AvatarFallback>
                          {contact.firstName?.[0]}{contact.lastName?.[0]}
                        </AvatarFallback>
                      </Avatar>
                      <div className="flex-1 min-w-0">
                        <h4 className="font-medium truncate">
                          {contact.firstName} {contact.lastName}
                        </h4>
                        <p className="text-sm text-gray-600 dark:text-gray-400 truncate">
                          {contact.title || 'No title'}
                        </p>
                        <div className="mt-2 space-y-1">
                          {contact.emailAddress && (
                            <div className="flex items-center text-xs text-gray-500">
                              <Mail className="w-3 h-3 mr-1" />
                              <span className="truncate">{contact.emailAddress}</span>
                            </div>
                          )}
                          {contact.deskPhone && (
                            <div className="flex items-center text-xs text-gray-500">
                              <Phone className="w-3 h-3 mr-1" />
                              <span>{contact.deskPhone}</span>
                            </div>
                          )}
                        </div>
                        <Badge variant="outline" className="mt-2 text-xs">
                          {contact.status}
                        </Badge>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
              
              {relatedData.contacts.length === 0 && (
                <div className="col-span-full">
                  <Card>
                    <CardContent className="p-8 text-center">
                      <Users className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                      <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2">
                        No contacts found
                      </h3>
                      <p className="text-gray-600 dark:text-gray-400 mb-4">
                        Start building relationships by adding contacts to this account.
                      </p>
                      <Button>
                        <Plus className="w-4 h-4 mr-2" />
                        Add First Contact
                      </Button>
                    </CardContent>
                  </Card>
                </div>
              )}
            </div>
          </TabsContent>

          <TabsContent value="deals" className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold">Active Deals & Opportunities</h3>
              <div className="flex items-center space-x-2">
                <Button variant="outline" size="sm">
                  <Filter className="w-4 h-4 mr-2" />
                  Filter
                </Button>
                <Button size="sm">
                  <Plus className="w-4 h-4 mr-2" />
                  New Deal
                </Button>
              </div>
            </div>
            
            <div className="space-y-4">
              {relatedData.deals.map((deal) => (
                <Card key={deal.id}>
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between">
                      <div className="flex-1">
                        <div className="flex items-center space-x-3">
                          <h4 className="font-medium">{deal.dealName}</h4>
                          <Badge variant={deal.stage === 'Order Won' ? 'default' : 'outline'}>
                            {deal.stage}
                          </Badge>
                        </div>
                        <div className="mt-2 grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                          <div>
                            <p className="text-gray-600 dark:text-gray-400">Value</p>
                            <p className="font-medium">{deal.dealValue || 'Not specified'}</p>
                          </div>
                          <div>
                            <p className="text-gray-600 dark:text-gray-400">Owner</p>
                            <p className="font-medium">{deal.dealOwner || 'Unassigned'}</p>
                          </div>
                          <div>
                            <p className="text-gray-600 dark:text-gray-400">Closing Date</p>
                            <p className="font-medium">
                              {deal.closingDate ? format(new Date(deal.closingDate), 'MMM d, yyyy') : 'TBD'}
                            </p>
                          </div>
                          <div>
                            <p className="text-gray-600 dark:text-gray-400">Probability</p>
                            <p className="font-medium">{deal.probability || 'Not set'}</p>
                          </div>
                        </div>
                        {deal.description && (
                          <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">
                            {deal.description}
                          </p>
                        )}
                      </div>
                      <div className="ml-4">
                        <Button variant="ghost" size="sm">
                          <ChevronRight className="w-4 h-4" />
                        </Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
              
              {relatedData.deals.length === 0 && (
                <Card>
                  <CardContent className="p-8 text-center">
                    <Target className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                    <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2">
                      No deals found
                    </h3>
                    <p className="text-gray-600 dark:text-gray-400 mb-4">
                      Create your first opportunity to start tracking potential revenue.
                    </p>
                    <Button>
                      <Plus className="w-4 h-4 mr-2" />
                      Create First Deal
                    </Button>
                  </CardContent>
                </Card>
              )}
            </div>
          </TabsContent>

          <TabsContent value="activities" className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold">Activity Timeline</h3>
              <div className="flex items-center space-x-2">
                <Button variant="outline" size="sm">
                  <Filter className="w-4 h-4 mr-2" />
                  Filter
                </Button>
                <Button size="sm">
                  <Plus className="w-4 h-4 mr-2" />
                  Log Activity
                </Button>
              </div>
            </div>
            
            <Card>
              <CardContent className="p-4">
                <ScrollArea className="h-[500px]">
                  <div className="space-y-4">
                    {relatedData.activities.map((activity, index) => (
                      <div key={activity.id} className="flex items-start space-x-4">
                        <div className="flex flex-col items-center">
                          <div className="p-2 bg-blue-100 dark:bg-blue-900 rounded-full">
                            {activity.activityType === 'Call' && <Phone className="w-4 h-4 text-blue-600" />}
                            {activity.activityType === 'Email' && <Mail className="w-4 h-4 text-blue-600" />}
                            {activity.activityType === 'LinkedIn Msg' && <Activity className="w-4 h-4 text-blue-600" />}
                            {!['Call', 'Email', 'LinkedIn Msg'].includes(activity.activityType) && <Activity className="w-4 h-4 text-blue-600" />}
                          </div>
                          {index < relatedData.activities.length - 1 && (
                            <div className="w-px h-8 bg-gray-200 dark:bg-gray-700 mt-2" />
                          )}
                        </div>
                        <div className="flex-1 pb-4">
                          <div className="flex items-center justify-between">
                            <h4 className="font-medium">{activity.activityType}</h4>
                            <time className="text-sm text-gray-500">
                              {format(new Date(activity.dateTime), 'MMM d, yyyy h:mm a')}
                            </time>
                          </div>
                          {activity.summary && (
                            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                              {activity.summary}
                            </p>
                          )}
                          {activity.outcomeDisposition && (
                            <Badge variant="outline" className="mt-2 text-xs">
                              {activity.outcomeDisposition}
                            </Badge>
                          )}
                          {activity.followUpSchedule && (
                            <div className="mt-2 text-xs text-gray-500 flex items-center">
                              <Calendar className="w-3 h-3 mr-1" />
                              Follow-up: {activity.followUpSchedule}
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                    
                    {relatedData.activities.length === 0 && (
                      <div className="text-center py-8">
                        <Activity className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                        <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2">
                          No activities recorded
                        </h3>
                        <p className="text-gray-600 dark:text-gray-400 mb-4">
                          Start logging interactions to build a comprehensive timeline.
                        </p>
                        <Button>
                          <Plus className="w-4 h-4 mr-2" />
                          Log First Activity
                        </Button>
                      </div>
                    )}
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="insights" className="space-y-4">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Account Health Analysis</CardTitle>
                  <CardDescription>
                    AI-powered insights based on account activity and engagement
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-sm">Engagement Level</span>
                      <Badge variant={relatedData.activities.length > 5 ? 'default' : 'secondary'}>
                        {relatedData.activities.length > 5 ? 'High' : relatedData.activities.length > 2 ? 'Medium' : 'Low'}
                      </Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm">Deal Pipeline</span>
                      <Badge variant={relatedData.deals.length > 0 ? 'default' : 'secondary'}>
                        {relatedData.deals.length > 0 ? 'Active' : 'No Deals'}
                      </Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm">Contact Coverage</span>
                      <Badge variant={relatedData.contacts.length > 2 ? 'default' : 'secondary'}>
                        {relatedData.contacts.length > 2 ? 'Good' : relatedData.contacts.length > 0 ? 'Limited' : 'None'}
                      </Badge>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Recommended Actions</CardTitle>
                  <CardDescription>
                    Suggested next steps to improve account performance
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {relatedData.contacts.length === 0 && (
                      <div className="p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
                        <p className="text-sm font-medium text-yellow-800 dark:text-yellow-200">
                          Add contacts to improve relationship mapping
                        </p>
                      </div>
                    )}
                    {relatedData.activities.length < 3 && (
                      <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                        <p className="text-sm font-medium text-blue-800 dark:text-blue-200">
                          Increase engagement with more touchpoints
                        </p>
                      </div>
                    )}
                    {relatedData.deals.length === 0 && (
                      <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                        <p className="text-sm font-medium text-green-800 dark:text-green-200">
                          Identify and create potential opportunities
                        </p>
                      </div>
                    )}
                    {relatedData.contacts.length > 0 && relatedData.activities.length > 2 && relatedData.deals.length > 0 && (
                      <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                        <p className="text-sm font-medium text-green-800 dark:text-green-200">
                          ✅ Account is well-managed with good engagement
                        </p>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
