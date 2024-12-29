# File: tests/extension/tests/setup.ts
import { test as base, BrowserContext, chromium, type Page } from '@playwright/test';
import path from 'path';

// Define custom fixture types
interface ExtensionFixtures {
  extensionContext: BrowserContext;
  backgroundPage: Page;
}

// Create test with extension fixtures
export const test = base.extend<ExtensionFixtures>({
  extensionContext: async ({ }, use) => {
    const extensionPath = path.join(__dirname, '../../extension');
    const context = await chromium.launchPersistentContext('', {
      headless: true,  // Required for CI
      args: [
        `--disable-extensions-except=${extensionPath}`,
        `--load-extension=${extensionPath}`,
        '--no-sandbox',  // Required for CI
        '--disable-setuid-sandbox',  // Required for CI
      ]
    });

    await use(context);
    await context.close();
  },

  backgroundPage: async ({ extensionContext }, use) => {
    // Wait for the background page to be available
    const backgroundPage = await extensionContext.waitForEvent('backgroundpage');
    await use(backgroundPage);
  },
});

export { expect } from '@playwright/test';
