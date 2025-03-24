# Deploying crawl4AI-agent to Railway

This guide explains how to deploy the crawl4AI-agent application to Railway.

## Prerequisites

- A [Railway](https://railway.app/) account
- Git installed on your local machine
- Your code pushed to a GitHub repository

## Deployment Steps

1. **Install the Railway CLI** (optional but helpful)
   ```bash
   npm i -g @railway/cli
   ```

2. **Login to Railway**
   ```bash
   railway login
   ```

3. **Initialize Railway in your project**
   ```bash
   railway init
   ```

4. **Link to an existing project or create a new one**
   ```bash
   railway link
   # Or to create a new project
   railway project create
   ```

5. **Set up environment variables**
   
   Go to your Railway project dashboard, navigate to the "Variables" tab, and add the following environment variables:
   
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `SUPABASE_URL`: Your Supabase project URL
   - `SUPABASE_SERVICE_KEY`: Your Supabase service key
   - Any other environment variables your application needs

6. **Deploy your application**
   ```bash
   railway up
   ```

   Alternatively, you can connect your GitHub repository to Railway for automatic deployments.

## Automatic Deployment from GitHub

1. Go to your Railway dashboard and create a new project
2. Select "Deploy from GitHub repo"
3. Choose your repository
4. Configure your environment variables
5. Railway will automatically deploy your application

## Monitoring and Logs

- Access logs from the Railway dashboard
- Monitor application performance and resource usage
- Set up alerts for any issues

## Custom Domain (Optional)

1. Go to the "Settings" tab in your Railway project
2. Click on "Domains"
3. Add your custom domain
4. Update your DNS settings as instructed

## Troubleshooting

- Check application logs for errors
- Verify environment variables are set correctly
- Ensure your application is listening on the correct port (`$PORT` environment variable)
