import {
  Contact, Account, ActivityLog, ActiveDeal, Lead, UserProfile,
  ApiResponse, PaginatedResponse,
  CreateContactRequest, UpdateContactRequest,
  CreateAccountRequest, UpdateAccountRequest,
  CreateActivityRequest, UpdateActivityRequest,
  CreateDealRequest, UpdateDealRequest,
  CreateLeadRequest, UpdateLeadRequest,
  UpdateUserProfileRequest, ReportRequest
} from '@shared/models';

const API_BASE = '/api';

// Generic API helper
async function apiRequest<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  try {
    console.log(`Making API request to: ${API_BASE}${endpoint}`);

    const response = await fetch(`${API_BASE}${endpoint}`, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    });

    console.log(`API response status: ${response.status} for ${endpoint}`);
    console.log(`API response headers:`, Object.fromEntries(response.headers.entries()));

    // Check if response has a body and handle accordingly
    if (!response.ok) {
      let errorText = '';
      try {
        if (!response.bodyUsed) {
          errorText = await response.text();
        }
      } catch (e) {
        errorText = 'Unable to read error response';
      }
      console.error(`API Error Response: "${errorText}"`);
      console.error(`Response status: ${response.status}, statusText: "${response.statusText}"`);
      throw new Error(`API request failed: ${response.status} ${response.statusText}${errorText ? ' - ' + errorText : ''}`);
    }

    // For successful responses, check content type first
    const contentType = response.headers.get('content-type');

    // Handle empty responses (like DELETE operations)
    if (response.status === 204 || !contentType) {
      return {} as T;
    }

    // Handle JSON responses
    if (contentType.includes('application/json')) {
      if (!response.bodyUsed) {
        return await response.json();
      } else {
        throw new Error('Response body already consumed');
      }
    }

    // Handle non-JSON responses
    let responseText = '';
    if (!response.bodyUsed) {
      responseText = await response.text();
    }
    console.error(`Expected JSON but got: ${contentType}`, responseText.substring(0, 200));
    throw new Error(`Expected JSON response but got ${contentType}`);

  } catch (error) {
    console.error(`API request error for ${endpoint}:`, error);
    throw error;
  }
}

// Contacts API
export const contactsApi = {
  async getAll(params?: { page?: number; limit?: number; search?: string }): Promise<PaginatedResponse<Contact>> {
    const searchParams = new URLSearchParams();
    if (params?.page) searchParams.append('page', params.page.toString());
    if (params?.limit) searchParams.append('limit', params.limit.toString());
    if (params?.search) searchParams.append('search', params.search);
    
    return apiRequest<PaginatedResponse<Contact>>(`/contacts?${searchParams}`);
  },

  async getById(id: string): Promise<ApiResponse<Contact>> {
    return apiRequest<ApiResponse<Contact>>(`/contacts/${id}`);
  },

  async create(data: CreateContactRequest): Promise<ApiResponse<Contact>> {
    return apiRequest<ApiResponse<Contact>>('/contacts', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },

  async update(id: string, data: UpdateContactRequest): Promise<ApiResponse<Contact>> {
    return apiRequest<ApiResponse<Contact>>(`/contacts/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  },

  async delete(id: string): Promise<ApiResponse<void>> {
    return apiRequest<ApiResponse<void>>(`/contacts/${id}`, {
      method: 'DELETE',
    });
  },
};

// Accounts API
export const accountsApi = {
  async getAll(params?: { page?: number; limit?: number; search?: string }): Promise<PaginatedResponse<Account>> {
    const searchParams = new URLSearchParams();
    if (params?.page) searchParams.append('page', params.page.toString());
    if (params?.limit) searchParams.append('limit', params.limit.toString());
    if (params?.search) searchParams.append('search', params.search);
    
    return apiRequest<PaginatedResponse<Account>>(`/accounts?${searchParams}`);
  },

  async getById(id: string): Promise<ApiResponse<Account>> {
    return apiRequest<ApiResponse<Account>>(`/accounts/${id}`);
  },

  async create(data: CreateAccountRequest): Promise<ApiResponse<Account>> {
    return apiRequest<ApiResponse<Account>>('/accounts', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },

  async update(id: string, data: UpdateAccountRequest): Promise<ApiResponse<Account>> {
    return apiRequest<ApiResponse<Account>>(`/accounts/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  },

  async delete(id: string): Promise<ApiResponse<void>> {
    return apiRequest<ApiResponse<void>>(`/accounts/${id}`, {
      method: 'DELETE',
    });
  },
};

// Activities API
export const activitiesApi = {
  async getAll(params?: { page?: number; limit?: number }): Promise<PaginatedResponse<ActivityLog>> {
    const searchParams = new URLSearchParams();
    if (params?.page) searchParams.append('page', params.page.toString());
    if (params?.limit) searchParams.append('limit', params.limit.toString());

    return apiRequest<PaginatedResponse<ActivityLog>>(`/activities?${searchParams}`);
  },

  async getById(id: string): Promise<ApiResponse<ActivityLog>> {
    return apiRequest<ApiResponse<ActivityLog>>(`/activities/${id}`);
  },

  async create(data: CreateActivityRequest): Promise<ApiResponse<ActivityLog>> {
    return apiRequest<ApiResponse<ActivityLog>>('/activities', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },

  async update(id: string, data: UpdateActivityRequest): Promise<ApiResponse<ActivityLog>> {
    return apiRequest<ApiResponse<ActivityLog>>(`/activities/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  },

  async delete(id: string): Promise<ApiResponse<void>> {
    return apiRequest<ApiResponse<void>>(`/activities/${id}`, {
      method: 'DELETE',
    });
  },
};

// Deals API
export const dealsApi = {
  async getAll(params?: { page?: number; limit?: number }): Promise<PaginatedResponse<ActiveDeal>> {
    const searchParams = new URLSearchParams();
    if (params?.page) searchParams.append('page', params.page.toString());
    if (params?.limit) searchParams.append('limit', params.limit.toString());

    return apiRequest<PaginatedResponse<ActiveDeal>>(`/deals?${searchParams}`);
  },

  async getById(id: string): Promise<ApiResponse<ActiveDeal>> {
    return apiRequest<ApiResponse<ActiveDeal>>(`/deals/${id}`);
  },

  async create(data: CreateDealRequest): Promise<ApiResponse<ActiveDeal>> {
    return apiRequest<ApiResponse<ActiveDeal>>('/deals', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },

  async update(id: string, data: UpdateDealRequest): Promise<ApiResponse<ActiveDeal>> {
    return apiRequest<ApiResponse<ActiveDeal>>(`/deals/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  },

  async delete(id: string): Promise<ApiResponse<void>> {
    return apiRequest<ApiResponse<void>>(`/deals/${id}`, {
      method: 'DELETE',
    });
  },
};

// Leads API
export const leadsApi = {
  async getAll(params?: { page?: number; limit?: number }): Promise<PaginatedResponse<Lead>> {
    const searchParams = new URLSearchParams();
    if (params?.page) searchParams.append('page', params.page.toString());
    if (params?.limit) searchParams.append('limit', params.limit.toString());

    return apiRequest<PaginatedResponse<Lead>>(`/leads?${searchParams}`);
  },

  async getById(id: string): Promise<ApiResponse<Lead>> {
    return apiRequest<ApiResponse<Lead>>(`/leads/${id}`);
  },

  async create(data: CreateLeadRequest): Promise<ApiResponse<Lead>> {
    return apiRequest<ApiResponse<Lead>>('/leads', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },

  async update(id: string, data: UpdateLeadRequest): Promise<ApiResponse<Lead>> {
    return apiRequest<ApiResponse<Lead>>(`/leads/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  },

  async delete(id: string): Promise<ApiResponse<void>> {
    return apiRequest<ApiResponse<void>>(`/leads/${id}`, {
      method: 'DELETE',
    });
  },
};

// Reports API
export const reportsApi = {
  async generate(data: ReportRequest): Promise<ApiResponse<any>> {
    return apiRequest<ApiResponse<any>>('/reports/generate', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },

  async download(id: string): Promise<ApiResponse<any>> {
    return apiRequest<ApiResponse<any>>(`/reports/download/${id}`);
  },
};

// User Profile API
export const userProfileApi = {
  async getCurrent(): Promise<ApiResponse<UserProfile>> {
    return apiRequest<ApiResponse<UserProfile>>('/profile');
  },

  async updateCurrent(data: UpdateUserProfileRequest): Promise<ApiResponse<UserProfile>> {
    return apiRequest<ApiResponse<UserProfile>>('/profile', {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  },

  async getById(id: string): Promise<ApiResponse<UserProfile>> {
    return apiRequest<ApiResponse<UserProfile>>(`/profile/${id}`);
  },

  async update(id: string, data: UpdateUserProfileRequest): Promise<ApiResponse<UserProfile>> {
    return apiRequest<ApiResponse<UserProfile>>(`/profile/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  },
};

// Export all APIs
export const api = {
  contacts: contactsApi,
  accounts: accountsApi,
  activities: activitiesApi,
  deals: dealsApi,
  leads: leadsApi,
  reports: reportsApi,
  profile: userProfileApi,
};

export default api;
