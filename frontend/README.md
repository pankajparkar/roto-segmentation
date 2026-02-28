# Roto-Seg Frontend

React TypeScript frontend for the AI rotoscoping automation system.

## Setup

```bash
# Install dependencies (using pnpm)
pnpm install

# Start development server
pnpm dev

# Build for production
pnpm build
```

## Project Structure

```
frontend/
├── src/
│   ├── components/    # Reusable UI components
│   ├── pages/         # Page components
│   ├── hooks/         # Custom React hooks
│   ├── lib/           # Utilities and API client
│   ├── store/         # State management
│   ├── types/         # TypeScript types
│   ├── App.tsx        # Root component
│   └── main.tsx       # Entry point
├── public/            # Static assets
├── index.html         # HTML template
└── package.json       # Dependencies
```

## Available Scripts

```bash
pnpm dev          # Start dev server
pnpm build        # Build for production
pnpm preview      # Preview production build
pnpm test         # Run tests
pnpm lint         # Run linter
```

## Environment Variables

Create a `.env.local` file:

```
VITE_API_URL=http://localhost:8000
```
