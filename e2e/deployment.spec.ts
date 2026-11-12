import { expect, test, type Page } from '@playwright/test';
import { mockHf } from './hf-mock';

/** The Hardware panel's Advanced block (the Parallelism panel has its own Advanced too). */
function hardwareAdvanced(page: Page) {
  return page.locator('details.advanced', { has: page.getByTestId('offload-advanced') });
}

/**
 * Deployment knobs on mocked Hub fixtures: speculative decoding on the bundled default model (Qwen/Qwen3.8-27B,
 * which ships an MTP head) and routed-expert offload on a MoE (Qwen/Qwen3.8-Flash-Next, 512 experts). Each knob shows
 * its result annotation (the spec-dec speedup on the decode rate, the Host RAM tier bar) and survives a reload.
 */

/** Picks `option` in the custom picker named `label`. */
async function pick(page: Page, label: string, option: string) {
  await page.getByRole('combobox', { name: label, exact: true }).click();
  await page.getByRole('option', { name: option, exact: true }).click();
}

test('MTP speculative decoding shows the speedup on the decode rate and survives a reload', async ({
  page,
}) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await mockHf(page);
  await page.goto('/');
  await expect(page.getByTestId('decode-tps')).toHaveText(/[1-9]/);
  const before = await page.getByTestId('decode-tps').textContent();
  await pick(page, 'Speculative decoding', 'MTP (model head)');
  await expect(page).toHaveURL(/sd=mtp/);
  await expect(page.getByTestId('spec-dec')).toContainText(/spec-dec ×\d+\.\d\d/);
  await expect(page.getByTestId('decode-tps')).not.toHaveText(before ?? '');
  const after = await page.getByTestId('decode-tps').textContent();
  await page.reload();
  await expect(
    page.getByRole('combobox', { name: 'Speculative decoding', exact: true }),
  ).toHaveText(/MTP/);
  await expect(page.getByTestId('decode-tps')).toHaveText(after ?? '');
  // Off again: the URL and the rate are back to the defaults.
  await pick(page, 'Speculative decoding', 'Off');
  await expect(page).not.toHaveURL(/sd=/);
  await expect(page.getByTestId('decode-tps')).toHaveText(before ?? '');
  await expect(page.getByTestId('spec-dec')).toHaveCount(0);
  expect(errors).toEqual([]);
});

test('DP2 lookup sharing halves the device table memory and survives a reload', async ({
  page,
}) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await mockHf(page);
  await page.goto('/?m=Qwen/Qwen3.8-Flash-Next&n=2&dp=2&b=2');
  const table = page.getByTestId('memory-legend').locator('li').filter({ hasText: 'n-gram table' });
  // Fixture: 102,400,000,000 bytes, displayed with three significant digits; DP2 owns half.
  await expect(table).toHaveText('n-gram table 102GB');
  await hardwareAdvanced(page).locator(':scope > summary').click();
  await pick(page, 'Share lookup tables across DP', 'on');
  await expect(page).toHaveURL(/ltdp=1/);
  await expect(table).toHaveText('n-gram table 51.2GB');
  await expect(page.getByTestId('memory-tier')).toHaveCount(0);
  await page.reload();
  await expect(table).toHaveText('n-gram table 51.2GB');
  await hardwareAdvanced(page).locator(':scope > summary').click();
  await expect(page.getByRole('combobox', { name: 'Share lookup tables across DP' })).toHaveText(
    'on',
  );
  await pick(page, 'Share lookup tables across DP', 'off');
  await expect(page).not.toHaveURL(/ltdp=/);
  await expect(table).toHaveText('n-gram table 102GB');
  expect(errors).toEqual([]);
});

test('DSpark without a released drafter uses the hypothetical one, badged', async ({ page }) => {
  await mockHf(page);
  await page.goto('/');
  await expect(page.getByTestId('decode-tps')).toHaveText(/[1-9]/);
  await pick(page, 'Speculative decoding', 'DSpark');
  await expect(page.getByTestId('hypothetical-badge')).toBeVisible();
  await expect(page.getByTestId('spec-dec')).toBeVisible();
});

test('DFlash 2 loads its released drafter config from the Hub', async ({ page }) => {
  await mockHf(page);
  await page.goto('/?sd=dflash2');
  await expect(page.getByRole('combobox', { name: 'Drafter', exact: true })).toHaveValue(
    'Model default (incoai/Qwen3.8-27B-DFlash2)',
  );
  await expect(page.getByTestId('spec-dec')).toBeVisible();
  await expect(page.getByTestId('hypothetical-badge')).toHaveCount(0);
});

test('MTP k 7: the batch chart marks the break-even batch between two sweep batches; off has no marker', async ({
  page,
}) => {
  await mockHf(page);
  await page.goto('/?sd=mtp&sdk=7');
  const marker = page.getByTestId('specdec-break-even');
  await expect(marker).toBeVisible();
  const text = (await marker.locator('text').textContent()) ?? '';
  expect(text).toMatch(/^break-even batch \d+$/);
  const breakEven = Number(text.replace('break-even batch ', ''));
  // Strictly between two adjacent sweep batches (dp·2^k), i.e. not a sweep batch itself.
  const batches = (
    await page.locator('[data-chart="sweep"] svg > g > text[y="177"]').allTextContents()
  ).map(Number);
  const hi = batches.findIndex((b) => b > breakEven);
  expect(hi).toBeGreaterThan(0);
  expect(batches[hi - 1]).toBeLessThan(breakEven);
  await expect(page.getByTestId('sweep-legend-off')).toBeVisible();
  // A sweep point past the marker: spec-dec is slower there.
  const past = page.locator('[data-chart="sweep"] [data-chart-point]').nth(hi);
  await past.scrollIntoViewIfNeeded();
  await past.focus();
  const tooltip = page.locator('[data-chart="sweep"] [role="tooltip"]');
  await expect(tooltip).toContainText('without spec-dec');
  await expect(tooltip).toContainText('×0.');
  await expect(tooltip).toContainText('@ 8k');
  await pick(page, 'Speculative decoding', 'Off');
  await expect(page).not.toHaveURL(/sd=/);
  await expect(page.getByTestId('specdec-break-even')).toHaveCount(0);
  await expect(page.getByTestId('sweep-legend-off')).toHaveCount(0);
});

test('speculative decoding works with pipeline parallelism (PP 2 on two GPUs)', async ({
  page,
}) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await mockHf(page);
  await page.goto('/?n=2&pp=2&sd=mtp');
  await expect(
    page.getByRole('combobox', { name: 'Speculative decoding', exact: true }),
  ).toHaveText(/MTP \(model head\)/);
  await expect(page.getByTestId('spec-dec')).toBeVisible();
  await expect(page.getByTestId('decode-tps')).toHaveText(/[1-9]/);
  await expect(page.getByText(/not supported/)).toHaveCount(0);
  expect(errors).toEqual([]);
});

test('expert offload to host RAM shows the tier bar and survives a reload', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await mockHf(page);
  await page.goto('/?m=Qwen/Qwen3.8-Flash-Next');
  await expect(page.getByTestId('model-facts')).toContainText('qwen');
  await expect(page.getByTestId('ngram-params')).toContainText('n-gram tables');
  await expect(page.getByTestId('decode-tps')).toHaveText(/[1-9]/);
  await expect(page.getByTestId('memory-tier')).toHaveCount(0);
  await pick(page, 'MoE experts', 'Host RAM');
  await expect(page).toHaveURL(/ox=ram/);
  await expect(page.getByTestId('memory-tier').filter({ hasText: 'Host RAM' })).toBeVisible();
  const decode = await page.getByTestId('decode-tps').textContent();
  await page.reload();
  await expect(page.getByRole('combobox', { name: 'MoE experts', exact: true })).toHaveText(
    'Host RAM',
  );
  await expect(page.getByTestId('memory-tier').filter({ hasText: 'Host RAM' })).toBeVisible();
  await expect(page.getByTestId('decode-tps')).toHaveText(decode ?? '');
  expect(errors).toEqual([]);
});

/** The hero's decode rate as a number (tok/s), after checking it is quoted at the 8k decode context. */
async function decodeAt8k(page: Page): Promise<number> {
  await expect(page.getByTestId('decode-tps')).toHaveText(/[1-9]/);
  await expect(page.locator('.answer').filter({ hasText: 'Decode' })).toContainText('tok/s @ 8k');
  return Number((await page.getByTestId('decode-tps').textContent())?.replace(/,/g, ''));
}

/**
 * Qwen3.8-Flash-Next (48 MoE layers, 512 experts, top-10) at fp8 on one RTX 5090 with DDR5-5600 × 2 channels
 * (96 GB) and the default 990 PRO SSD; batch 1, decode @ 8k, resident fraction 0, PLE tables on the SSD in all three.
 *
 * Per decode token every MoE layer fetches its 10 routed experts, M · W = 10 × 3 · 2560 · 640 B ≈ 49 MB
 * (2.36 GB over the 48 layers). One layer's fetch costs (engine numbers when this was written):
 * - host RAM: ≈ 0.85 ms, mostly the PCIe 5.0 x16 link (≈ 63 GB/s; the RAM's 89.6 GB/s is higher) plus request latency;
 * - SSD: ≈ 6.64 ms, bounded by the drive's ≈ 7.45 GB/s sequential read (the host staging copy is faster).
 * Without overlap the step's transfer time would be the sum over layers: all RAM 48 × 0.85 ≈ 41 ms, all SSD
 * 48 × 6.64 ≈ 319 ms, and the 24/24 split their mean, ≈ 180 ms. The split reads from both host RAM and the SSD, so
 * the prefetched 80% of the two streams overlaps (only the experts known after the router stay serial): ≈ 164 ms per
 * token instead of the 180 ms mean. Prefetch (p = 0.8) also hides part of each fetch under the GPU work:
 * ≈ 23.8, 6.07 and 3.12 tok/s @ 8k. The split sits strictly between the extremes, and its time per token is below
 * the mean of the two.
 * Fit: all experts (≈ 121 GB) overflow the 96 GB of host RAM; the split puts ≈ 60 GB in RAM and ≈ 60 GB on the SSD.
 */
test('a 50/50 host-RAM + SSD expert split decodes between all-RAM and all-SSD', async ({
  page,
}) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await mockHf(page);
  const base = '/?m=Qwen/Qwen3.8-Flash-Next&w=fp8_e4m3&hr=ddr5-5600-2ch&olt=ssd';
  const hostRam = page.getByTestId('memory-tier').filter({ hasText: 'Host RAM' });
  const ssdTier = page.getByTestId('memory-tier').filter({ hasText: 'SSD' });
  await page.goto(`${base}&ox=ram`);
  const ram = await decodeAt8k(page);
  await expect(hostRam).toContainText('does not fit');
  await page.goto(`${base}&ox=split&oxr=24`);
  const split = await decodeAt8k(page);
  // Each tier bar shows its share of the experts, by layer span, and both fit.
  await expect(hostRam).toContainText('MoE experts, layers 0–23');
  await expect(ssdTier).toContainText('MoE experts, layers 24–47');
  await expect(hostRam).not.toContainText('does not fit');
  await expect(ssdTier).not.toContainText('does not fit');
  await expect(page.getByLabel('Layers in host RAM', { exact: true })).toHaveValue('24');
  await page.goto(`${base}&ox=ssd`);
  const ssd = await decodeAt8k(page);
  expect(ram).toBeGreaterThan(split);
  expect(split).toBeGreaterThan(ssd);
  // Overlap across tiers beats the serial mean: the split's time per token is below the mean of the two.
  expect(1 / split).toBeLessThan((1 / ram + 1 / ssd) / 2);
  expect(errors).toEqual([]);
});

test('picking Host RAM + SSD writes ox=split with half the layers in RAM, and survives a reload', async ({
  page,
}) => {
  await mockHf(page);
  await page.goto('/?m=Qwen/Qwen3.8-Flash-Next&olt=ssd');
  await expect(page.getByTestId('decode-tps')).toHaveText(/[1-9]/);
  await pick(page, 'MoE experts', 'Host RAM + SSD');
  await expect(page).toHaveURL(/ox=split&oxr=24/);
  await pick(page, 'Host RAM holds', 'Latest layers');
  await expect(page).toHaveURL(/ox=split&oxr=24&oxe=last/);
  await expect(page.getByTestId('memory-tier').filter({ hasText: 'Host RAM' })).toContainText(
    'MoE experts, layers 24–47',
  );
  await page.reload();
  await expect(page.getByRole('combobox', { name: 'MoE experts', exact: true })).toHaveText(
    'Host RAM + SSD',
  );
  await expect(page.getByRole('combobox', { name: 'Host RAM holds', exact: true })).toHaveText(
    'Latest layers',
  );
  await expect(page.getByLabel('Layers in host RAM', { exact: true })).toHaveValue('24');
});

test('every deployment and host URL key loads, stays canonical and survives a reload', async ({
  page,
}) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await mockHf(page);
  // Every new key with a non-default value, in the codec's canonical order (url.ts).
  const query =
    'm=Qwen/Qwen3.8-Flash-Next&hr=ddr5-6400-8ch&hrc=512&ssd=nvme-t705-2tb&hl=4x8' +
    '&ox=split&oxl=3&oxf=0.25&oxh=fine&oxr=20&oxe=last' +
    '&olt=split&ols=interleaved&olr=10&ole=last&oqd=32&opf=0' +
    '&sd=dflash2&sdr=hyp&sdk=5&sdu=code';
  await page.goto(`/?${query}`);
  await expect(page.getByTestId('decode-tps')).toHaveText(/[1-9]/);
  await expect(page.getByText('Ignored invalid URL values')).toHaveCount(0);
  await expect(page.getByTestId('spec-dec')).toBeVisible();
  await expect(page.getByTestId('memory-tier').filter({ hasText: 'SSD' })).toBeVisible();
  // The app rewrites the URL from its state (after a short debounce): it must come back unchanged.
  await page.waitForTimeout(400);
  expect(new URL(page.url()).search).toBe(`?${query}`);
  const decode = await page.getByTestId('decode-tps').textContent();
  await page.reload();
  await expect(page.getByTestId('decode-tps')).toHaveText(decode ?? '');
  await page.waitForTimeout(400);
  expect(new URL(page.url()).search).toBe(`?${query}`);
  // A custom acceptance replaces the use case in the URL.
  await page.goto(`/?${query.replace('sdu=code', 'sda=0.8')}`);
  await expect(page.getByRole('combobox', { name: 'Use case', exact: true })).toHaveText('Custom');
  await expect(page.getByLabel('Acceptance α', { exact: true })).toHaveValue('0.8');
  expect(errors).toEqual([]);
});

test('oet=ram places the token embeddings in host RAM: a tier bar segment, a chip, and the URL round-trips', async ({
  page,
}) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await mockHf(page);
  await page.goto('/?m=Qwen/Qwen3.8-27B&oet=ram');
  await expect(page.getByTestId('decode-tps')).toHaveText(/[1-9]/);
  await expect(page.getByText('Ignored invalid URL values')).toHaveCount(0);
  await expect(page.getByTestId('memory-tier').filter({ hasText: 'Host RAM' })).toContainText(
    'Token embeddings',
  );
  await expect(page.getByText('Token embeddings: host RAM')).toBeVisible();
  await page.waitForTimeout(400);
  // Qwen3.8-27B is the default model, so `m` is not written.
  expect(new URL(page.url()).search).toBe('?oet=ram');
  await page.reload();
  await expect(page.getByTestId('decode-tps')).toHaveText(/[1-9]/);
  await hardwareAdvanced(page).locator('> summary').click();
  await expect(
    page.getByTestId('offload-advanced').getByRole('combobox', { name: 'Token embeddings' }),
  ).toHaveText('Host RAM');
  expect(errors).toEqual([]);
});

test('a legacy offload-overlap URL is rewritten to the equivalent prefetch accuracy', async ({
  page,
}) => {
  await mockHf(page);
  await page.goto('/?m=Qwen/Qwen3.8-Flash-Next&oo=none&ox=ram');
  await expect(page.getByTestId('decode-tps')).toHaveText(/[1-9]/);
  await expect(page.getByText('Ignored invalid URL values')).toHaveCount(0);
  await page.waitForTimeout(400);
  expect(new URL(page.url()).search).toBe('?m=Qwen/Qwen3.8-Flash-Next&ox=ram&opf=0');
  await hardwareAdvanced(page).locator('> summary').click();
  await expect(page.getByTestId('offload-advanced')).toBeVisible();
  await expect(page.getByLabel('Prefetch accuracy %', { exact: true })).toHaveValue('0');
});

test('an old whole-layer offload URL (odl / odt, removed) loads without a notice and drops those keys', async ({
  page,
}) => {
  await mockHf(page);
  await page.goto('/?m=Qwen/Qwen3.8-Flash-Next&ox=ram&odl=40&odt=ssd');
  await expect(page.getByTestId('decode-tps')).toHaveText(/[1-9]/);
  await expect(page.getByText('Ignored invalid URL values')).toHaveCount(0);
  await page.waitForTimeout(400);
  expect(new URL(page.url()).search).toBe('?m=Qwen/Qwen3.8-Flash-Next&ox=ram');
  await expect(page.getByRole('combobox', { name: 'Dense layers' })).toHaveCount(0);
});

test('Host RAM picker groups DDR5 before DDR4, searches, and a DDR4 id round-trips in the URL', async ({
  page,
}) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await mockHf(page);
  await page.goto('/?m=Qwen/Qwen3.8-Flash-Next&ox=ram');
  await page.getByRole('combobox', { name: 'Host RAM', exact: true }).click();
  const groups = page.getByRole('listbox').getByRole('group');
  await expect(groups.nth(0)).toContainText('DDR5');
  await expect(groups.nth(1)).toContainText('DDR4');
  await page.screenshot({ path: 'test-results/host-ram-picker.png' });
  await page.getByRole('combobox', { name: 'Host RAM', exact: true }).press('Escape');
  await page.getByRole('combobox', { name: 'Host RAM', exact: true }).click();
  await page.keyboard.type('ddr4 3200 8');
  await page.getByRole('option', { name: /DDR4-3200, 8 channels/ }).click();
  await expect(page).toHaveURL(/hr=ddr4-3200-8ch/);
  await page.reload();
  await expect(page.getByRole('combobox', { name: 'Host RAM', exact: true })).toHaveText(
    /DDR4-3200, 8 channels/,
  );
  expect(errors).toEqual([]);
});

for (const viewport of [
  { width: 1440, height: 1000 },
  { width: 390, height: 844 },
]) {
  test(`SSD read rate picker persists and only appears for SSD offload (${viewport.width}px)`, async ({
    page,
  }) => {
    await page.setViewportSize(viewport);
    await mockHf(page);
    await page.goto('/?m=Qwen/Qwen3.8-Flash-Next&ox=ssd');
    if (viewport.width < 600)
      await page.locator('.panel > summary').filter({ hasText: 'Hardware' }).click();
    await hardwareAdvanced(page).locator('> summary').click();
    const rate = page.getByRole('combobox', { name: 'SSD read rate', exact: true });
    await expect(rate).toHaveText('Datasheet');
    await expect(page).not.toHaveURL(/osr=/);
    await pick(page, 'SSD read rate', 'Sustained');
    await expect(page).toHaveURL(/osr=sustained/);
    await page.reload();
    if (viewport.width < 600)
      await page.locator('.panel > summary').filter({ hasText: 'Hardware' }).click();
    await hardwareAdvanced(page).locator('> summary').click();
    await expect(rate).toHaveText('Sustained');
    await rate.scrollIntoViewIfNeeded();
    if (process.env.SSD_SCREENSHOTS_DIR) {
      await page.screenshot({
        path: `${process.env.SSD_SCREENSHOTS_DIR}/ssd-rate-${viewport.width}.png`,
      });
      await page.emulateMedia({ colorScheme: 'dark' });
      await page.screenshot({
        path: `${process.env.SSD_SCREENSHOTS_DIR}/ssd-rate-${viewport.width}-dark.png`,
      });
    }
    await pick(page, 'SSD read rate', 'Datasheet');
    await expect(page).not.toHaveURL(/osr=/);
    await page.goto('/?m=Qwen/Qwen3.8-Flash-Next&ox=ram');
    if (viewport.width < 600)
      await page.locator('.panel > summary').filter({ hasText: 'Hardware' }).click();
    await hardwareAdvanced(page).locator('> summary').click();
    await expect(rate).toHaveCount(0);
  });
}
