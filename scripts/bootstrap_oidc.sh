#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# One-time setup: lets GitHub Actions authenticate to AWS via OIDC
# (short-lived tokens — no AWS access keys stored in GitHub).
#
# Run once from CloudShell or any shell with admin credentials:
#   ./scripts/bootstrap_oidc.sh <github-owner>/<repo-name> [region]
#
# Then add the printed role ARN as the GitHub secret AWS_DEPLOY_ROLE_ARN.
# Safe to re-run: existing provider/role are reused and the policy updated.
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO="${1:?Usage: $0 <owner>/<repo> [region]}"
REGION="${2:-us-east-1}"
ROLE_NAME="GitHubActionsAdultIncomeDeployRole"
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
BUCKET="sagemaker-adult-income-cdk-${ACCOUNT}"
PROVIDER_ARN="arn:aws:iam::${ACCOUNT}:oidc-provider/token.actions.githubusercontent.com"

echo "Account : ${ACCOUNT}"
echo "Repo    : ${REPO}"
echo "Region  : ${REGION}"

# ─── 1. OIDC identity provider (one per account) ─────────────────────
if aws iam get-open-id-connect-provider --open-id-connect-provider-arn "$PROVIDER_ARN" >/dev/null 2>&1; then
  echo "✅ OIDC provider exists"
else
  aws iam create-open-id-connect-provider \
    --url https://token.actions.githubusercontent.com \
    --client-id-list sts.amazonaws.com \
    --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1
  echo "✅ OIDC provider created"
fi

# ─── 2. Role trusted ONLY by this repository ─────────────────────────
TRUST=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": { "Federated": "${PROVIDER_ARN}" },
    "Action": "sts:AssumeRoleWithWebIdentity",
    "Condition": {
      "StringEquals": { "token.actions.githubusercontent.com:aud": "sts.amazonaws.com" },
      "StringLike":   { "token.actions.githubusercontent.com:sub": "repo:${REPO}:*" }
    }
  }]
}
EOF
)

if aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
  aws iam update-assume-role-policy --role-name "$ROLE_NAME" --policy-document "$TRUST"
  echo "✅ Role exists — trust policy updated"
else
  aws iam create-role --role-name "$ROLE_NAME" \
    --assume-role-policy-document "$TRUST" \
    --description "GitHub Actions OIDC role for ${REPO}" >/dev/null
  echo "✅ Role created"
fi

# ─── 3. Least-privilege permissions for the 3 pipelines ──────────────
POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "CdkDeployViaBootstrapRoles",
      "Effect": "Allow",
      "Action": ["sts:AssumeRole", "iam:PassRole"],
      "Resource": "arn:aws:iam::${ACCOUNT}:role/cdk-*"
    },
    {
      "Sid": "CdkSynthLookups",
      "Effect": "Allow",
      "Action": ["ec2:DescribeVpcs", "ec2:DescribeSubnets"],
      "Resource": "*"
    },
    {
      "Sid": "StackStatusAndOutputs",
      "Effect": "Allow",
      "Action": ["cloudformation:DescribeStacks", "cloudformation:DeleteStack"],
      "Resource": "arn:aws:cloudformation:${REGION}:${ACCOUNT}:stack/AdultIncomeSageMakerStack/*"
    },
    {
      "Sid": "ProjectBucket",
      "Effect": "Allow",
      "Action": ["s3:CreateBucket", "s3:PutBucketPublicAccessBlock", "s3:ListBucket",
                 "s3:GetBucketLocation", "s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
      "Resource": ["arn:aws:s3:::${BUCKET}", "arn:aws:s3:::${BUCKET}/*"]
    },
    {
      "Sid": "SageMakerPipelineRegistryEndpoint",
      "Effect": "Allow",
      "Action": "sagemaker:*",
      "Resource": "arn:aws:sagemaker:${REGION}:${ACCOUNT}:*"
    },
    {
      "Sid": "SageMakerListCalls",
      "Effect": "Allow",
      "Action": ["sagemaker:List*", "sagemaker:Search"],
      "Resource": "*"
    },
    {
      "Sid": "PassSageMakerExecutionRole",
      "Effect": "Allow",
      "Action": "iam:PassRole",
      "Resource": "arn:aws:iam::${ACCOUNT}:role/AdultIncomeSageMakerExecutionRole",
      "Condition": { "StringEquals": { "iam:PassedToService": "sagemaker.amazonaws.com" } }
    }
  ]
}
EOF
)

aws iam put-role-policy --role-name "$ROLE_NAME" \
  --policy-name AdultIncomePipelines --policy-document "$POLICY"
echo "✅ Permissions attached"

echo
echo "────────────────────────────────────────────────────────────"
echo "Add this as GitHub secret AWS_DEPLOY_ROLE_ARN:"
aws iam get-role --role-name "$ROLE_NAME" --query Role.Arn --output text
echo "────────────────────────────────────────────────────────────"
