-- CreateEnum
CREATE TYPE "public"."ContactSource" AS ENUM ('Data Research', 'Referral', 'Event');

-- CreateEnum
CREATE TYPE "public"."ContactStatus" AS ENUM ('Suspect', 'Prospect', 'Active Deal', 'Do Not Call');

-- CreateEnum
CREATE TYPE "public"."AccountRating" AS ENUM ('Platinum (Must Have)', 'Gold (High Priority)', 'Silver (Medium Priority)', 'Bronze (Low Priority)');

-- CreateEnum
CREATE TYPE "public"."AccountStatus" AS ENUM ('Suspect', 'Prospect', 'Active Deal', 'Do Not Call');

-- CreateEnum
CREATE TYPE "public"."ActivityType" AS ENUM ('Call', 'Email', 'LinkedIn Msg', 'SMS', 'Other');

-- CreateEnum
CREATE TYPE "public"."OutcomeDisposition" AS ENUM ('Voicemail', 'RNR', 'Meeting Fixed', 'Meeting Completed', 'Meeting Rescheduled', 'Not Interested', 'Do not Call', 'Callback requested', 'Email sent', 'Email Received');

-- CreateEnum
CREATE TYPE "public"."BusinessLine" AS ENUM ('Human Capital', 'Managed Services', 'GCC', 'Automation', 'Support', 'Product', 'Solution', 'RCM');

-- CreateEnum
CREATE TYPE "public"."Geography" AS ENUM ('Americas', 'India', 'Philippines', 'EMEA', 'ANZ');

-- CreateEnum
CREATE TYPE "public"."Entity" AS ENUM ('Yitro Global', 'Yitro Tech');

-- CreateEnum
CREATE TYPE "public"."DealStage" AS ENUM ('Opportunity Identified', 'Proposal Submitted', 'Negotiating', 'Closing', 'Order Won', 'Order Lost');

-- CreateEnum
CREATE TYPE "public"."LeadSource" AS ENUM ('Website', 'Referral', 'Trade Show', 'Cold Call', 'Email', 'Partner');

-- CreateEnum
CREATE TYPE "public"."LeadStatus" AS ENUM ('New', 'Working', 'Qualified', 'Unqualified');

-- CreateEnum
CREATE TYPE "public"."LeadRating" AS ENUM ('Hot', 'Warm', 'Cold');

-- CreateTable
CREATE TABLE "public"."contacts" (
    "id" TEXT NOT NULL,
    "firstName" TEXT NOT NULL,
    "lastName" TEXT NOT NULL,
    "title" TEXT,
    "associatedAccount" TEXT,
    "emailAddress" TEXT,
    "deskPhone" TEXT,
    "mobilePhone" TEXT,
    "city" TEXT,
    "state" TEXT,
    "country" TEXT,
    "timeZone" TEXT,
    "source" "public"."ContactSource",
    "owner" TEXT,
    "status" "public"."ContactStatus",
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "createdBy" TEXT NOT NULL,
    "updatedBy" TEXT NOT NULL,

    CONSTRAINT "contacts_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."accounts" (
    "id" TEXT NOT NULL,
    "accountName" TEXT NOT NULL,
    "accountRating" "public"."AccountRating",
    "accountOwner" TEXT,
    "status" "public"."AccountStatus",
    "industry" TEXT,
    "revenue" TEXT,
    "numberOfEmployees" TEXT,
    "addressLine1" TEXT,
    "addressLine2" TEXT,
    "city" TEXT,
    "state" TEXT,
    "country" TEXT,
    "zipPostCode" TEXT,
    "timeZone" TEXT,
    "boardNumber" TEXT,
    "website" TEXT,
    "geo" "public"."Geography",
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "createdBy" TEXT NOT NULL,
    "updatedBy" TEXT NOT NULL,

    CONSTRAINT "accounts_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."activity_logs" (
    "id" TEXT NOT NULL,
    "activityType" "public"."ActivityType" NOT NULL,
    "associatedContact" TEXT,
    "associatedAccount" TEXT,
    "dateTime" TIMESTAMP(3) NOT NULL,
    "followUpSchedule" TEXT,
    "summary" TEXT,
    "outcomeDisposition" "public"."OutcomeDisposition",
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "createdBy" TEXT NOT NULL,
    "updatedBy" TEXT NOT NULL,

    CONSTRAINT "activity_logs_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."active_deals" (
    "id" TEXT NOT NULL,
    "dealOwner" TEXT,
    "dealName" TEXT NOT NULL,
    "businessLine" "public"."BusinessLine",
    "associatedAccount" TEXT,
    "associatedContact" TEXT,
    "closingDate" TIMESTAMP(3),
    "probability" TEXT,
    "dealValue" TEXT,
    "approvedBy" TEXT,
    "description" TEXT,
    "nextStep" TEXT,
    "geo" "public"."Geography",
    "entity" "public"."Entity",
    "stage" "public"."DealStage",
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "createdBy" TEXT NOT NULL,
    "updatedBy" TEXT NOT NULL,

    CONSTRAINT "active_deals_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."leads" (
    "id" TEXT NOT NULL,
    "firstName" TEXT NOT NULL,
    "lastName" TEXT NOT NULL,
    "company" TEXT NOT NULL,
    "title" TEXT,
    "phone" TEXT,
    "email" TEXT,
    "leadSource" "public"."LeadSource",
    "status" "public"."LeadStatus",
    "rating" "public"."LeadRating",
    "owner" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "createdBy" TEXT NOT NULL,
    "updatedBy" TEXT NOT NULL,

    CONSTRAINT "leads_pkey" PRIMARY KEY ("id")
);

-- AddForeignKey
ALTER TABLE "public"."contacts" ADD CONSTRAINT "contacts_associatedAccount_fkey" FOREIGN KEY ("associatedAccount") REFERENCES "public"."accounts"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."activity_logs" ADD CONSTRAINT "activity_logs_associatedContact_fkey" FOREIGN KEY ("associatedContact") REFERENCES "public"."contacts"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."activity_logs" ADD CONSTRAINT "activity_logs_associatedAccount_fkey" FOREIGN KEY ("associatedAccount") REFERENCES "public"."accounts"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."active_deals" ADD CONSTRAINT "active_deals_associatedAccount_fkey" FOREIGN KEY ("associatedAccount") REFERENCES "public"."accounts"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."active_deals" ADD CONSTRAINT "active_deals_associatedContact_fkey" FOREIGN KEY ("associatedContact") REFERENCES "public"."contacts"("id") ON DELETE SET NULL ON UPDATE CASCADE;
