#!/bin/bash

# Deploy ŚRUTI to Google Cloud Run
# Usage: ./deploy/cloud-run-deploy.sh [project-id]

set -e

PROJECT_ID="${1:-avyay-platform}"
REGION="asia-southeast1"
SERVICE="shruti"
IMAGE="asia-southeast1-docker.pkg.dev/$PROJECT_ID/avyay/$SERVICE"

echo "🕉️  Deploying ŚRUTI (श्रुति) to Google Cloud Run"
echo "📁 Project: $PROJECT_ID"
echo "🌏 Region: $REGION"
echo "🔧 Service: $SERVICE"
echo "📦 Image: $IMAGE"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found. Please install Google Cloud SDK."
    exit 1
fi

# Check if docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker."
    exit 1
fi

# Set project
echo "🔧 Setting project..."
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "🔌 Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com

# Create Artifact Registry repository if it doesn't exist
echo "📦 Creating Artifact Registry repository..."
gcloud artifacts repositories create avyay \
    --repository-format=docker \
    --location=$REGION \
    --description="Avyay AI Docker images" \
    2>/dev/null || echo "Repository already exists"

# Configure Docker authentication
echo "🔐 Configuring Docker authentication..."
gcloud auth configure-docker asia-southeast1-docker.pkg.dev

# Build the image
echo "🔨 Building Docker image..."
docker build -t $IMAGE .

# Push the image
echo "📤 Pushing image to Artifact Registry..."
docker push $IMAGE

# Deploy to Cloud Run
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy $SERVICE \
    --image $IMAGE \
    --region $REGION \
    --platform managed \
    --memory 512Mi \
    --cpu 1 \
    --min-instances 0 \
    --max-instances 3 \
    --port 8000 \
    --set-env-vars "CHROMA_PERSIST_DIR=/data/knowledge_base" \
    --set-secrets "GROQ_API_KEY=groq-api-key:latest" \
    --allow-unauthenticated \
    --timeout 300

# Get the service URL
SERVICE_URL=$(gcloud run services describe $SERVICE --region=$REGION --format="value(status.url)")

echo ""
echo "✅ Deployment completed successfully!"
echo "🌐 Service URL: $SERVICE_URL"
echo "📚 API Docs: $SERVICE_URL/docs"
echo "🔍 Health Check: $SERVICE_URL/health"
echo ""
echo "📝 Next steps:"
echo "   1. Test the health endpoint: curl $SERVICE_URL/health"
echo "   2. Visit the API docs: $SERVICE_URL/docs"
echo "   3. Test transcription: POST $SERVICE_URL/api/transcribe"