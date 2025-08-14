# Yitro CRM Platform

## Technology Stack

### Frontend

- **React 18** with TypeScript
- **Vite** for fast development and building
- **TailwindCSS** for styling
- **Radix UI** for component primitives
- **React Router** for navigation

### Backend

- **Express.js** RESTful API
- **Prisma ORM** for database management
- **Neon PostgreSQL** for data persistence
- **TypeScript** for type safety

### Infrastructure

- **Production-ready** architecture
- **Database migrations** for schema management
- **API validation** and error handling
- **Build optimization** for deployment

##  Installation & Setup

### Prerequisites

- Node.js 18+
- npm or yarn package manager

### Quick Start

1. **Clone & Install**

   ```bash
   git clone <repository-url>
   cd yitro-crm
   npm install
   ```

2. **Database Setup**
   The CRM is pre-configured with a Neon PostgreSQL database. The connection is already established and migrations will run automatically.

3. **Start Development**

   ```bash
   npm run dev
   ```

   The application will be available at `http://localhost:8080`

4. **Build for Production**
   ```bash
   npm run build
   ```

##  Usage Guide

### Getting Started

1. **Login**: Use the professional login interface (click "Login" on the landing page)
2. **Dashboard**: Overview of key metrics and recent activities
3. **Navigation**: Use the top navigation tabs to access different modules

### Profile & Settings

- Click the **Settings icon** or user avatar in the header
- Update personal information, contact details, and profile photo
- Configure notification preferences
- Set timezone and language preferences

### CRM Modules

#### Leads Management

- **View All Leads**: Comprehensive list with search and filtering
- **Add New Lead**: Complete lead capture form
- **Lead Details**: Full contact information and qualification status
- **Lead Conversion**: Convert qualified leads to active deals

#### Contacts Management

- **Contact List**: Professional contact directory
- **Contact Details**: Complete contact profiles with relationship tracking
- **Activities**: Track all interactions with contacts
- **Account Association**: Link contacts to their organizations

#### Accounts Management

- **Account Directory**: Company and organization database
- **Account Profiles**: Detailed company information and hierarchies
- **Relationship Mapping**: Contact associations and deal history
- **Territory Management**: Geographic and ownership assignments

#### Active Deals Management

- **Sales Pipeline**: Visual pipeline with stage tracking
- **Deal Management**: Complete opportunity lifecycle
- **Revenue Forecasting**: Pipeline value and probability tracking
- **Close Date Management**: Timeline and milestone tracking

#### Activities & Tasks

- **Activity Logging**: Calls, emails, meetings, and notes
- **Task Management**: Follow-ups and action items
- **Calendar Integration**: Schedule and timeline management
- **Outcome Tracking**: Results and next steps

#### Reports & Analytics

- **Sales Performance**: Revenue and quota tracking
- **Activity Reports**: Communication and engagement metrics
- **Pipeline Analysis**: Opportunity and forecast reports
- **Export Capabilities**: Excel and PDF downloads


## Configuration

### Environment Variables

The application uses secure environment variables for database connections. These are pre-configured for the included Neon database.

### Customization

- **Branding**: Update logos and company information
- **User Roles**: Configure access permissions
- **Business Logic**: Customize sales processes and workflows
- **Integration**: Connect with external systems

##  Data Structure

### Core Entities

- **Contacts**: Individual people and their information
- **Accounts**: Companies and organizations
- **Leads**: Potential customers and prospects
- **Active Deals**: Sales deals and pipeline management
- **Activities**: Interactions and communications
- **Users**: System users and profiles

### Relationships

- Contacts belong to Accounts
- Active Deals are associated with Accounts and Contacts
- Activities track interactions across all entities
- Users own and manage records

##  Security & Privacy

### Data Protection

- **Secure Database**: Encrypted PostgreSQL database
- **Input Validation**: Comprehensive data validation
- **Error Handling**: Secure error management
- **Access Control**: User-based permissions

### Best Practices

- Regular security updates
- Data backup procedures
- User access monitoring
- Compliance preparation

##  Deployment

### Production Build

```bash
npm run build
```

### Deployment Options

- **Netlify**: Automatic deployment from Git
- **Vercel**: Edge deployment with global CDN
- **AWS/GCP/Azure**: Cloud platform deployment
- **Docker**: Containerized deployment

### Development

- **TypeScript**: Full type safety throughout
- **Component Architecture**: Reusable and maintainable
- **API Design**: RESTful and consistent
- **Testing Ready**: Structured for test implementation


