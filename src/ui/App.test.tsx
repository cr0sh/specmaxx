// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { App, type ModelLoader } from './App';
import { inlineRunner } from './sim/runner';
import { FAMILIES, mapHfModel, type FamilyEntry } from '../sim/models/hf';
import { REF_DENSE_8B } from '../sim/models/reference';
import {
  fixtureRepoFiles,
  fixtureStem,
  readFixtureJson,
} from '../sim/models/__fixtures__/fixtures.testutil';
import { createCache } from '../hf/cache';
import { isHfError } from '../hf/errors';
import { fetchHfRepoFiles } from '../hf/fetch';
import { memoryStorage } from '../hf/storage';
import { HF_TOKEN_KEY } from '../hf/token';
import { apiUrl, resolveUrl } from '../hf/urls';
import { decodeState, encodeState } from './state';

const fixtureLoader: ModelLoader = async (id) => mapHfModel(fixtureRepoFiles(id));
function mount(loader = fixtureLoader) {
  return render(<App loader={loader} runner={inlineRunner} />);
}
beforeEach(() => {
  history.replaceState(null, '', '/');
  localStorage.clear();
});
afterEach(cleanup);

// Full App mounts of a 72-device MoE run, several times each: ~5 s on a fast laptop, so slow
// 2-4 vCPU CI runners need far more than vitest's 5 s default.
const HEAVY_APP_TEST_TIMEOUT_MS = 60_000;

describe('quant picker defaults', () => {
  it('labels the default option with the resolved format and updates when the model changes', async () => {
    history.replaceState(null, '', '/?m=Qwen/Qwen3.8-27B');
    mount(async (id) =>
      mapHfModel(fixtureRepoFiles(id, id.includes('EXL3') ? { dir: 'quant' } : {})),
    );
    await screen.findByText(/qwen:/);
    await waitFor(() =>
      expect(screen.getByRole('combobox', { name: 'Weights' }).textContent).toMatch(
        /Model default \(bf16\)/,
      ),
    );
    expect(screen.getByRole('combobox', { name: 'KV cache' }).textContent).toMatch(
      /Model default \(bf16\)/,
    );
    const input = screen.getByRole('combobox', { name: 'Hugging Face model ID' });
    fireEvent.change(input, {
      target: { value: 'bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-2.5bpw' },
    });
    fireEvent.keyDown(input, { key: 'Enter' });
    await waitFor(() =>
      expect(screen.getByRole('combobox', { name: 'Weights' }).textContent).toMatch(
        /Model default \(exl3 2\.52bpw\)/,
      ),
    );
    fireEvent.click(screen.getByRole('combobox', { name: 'Weights' }));
    expect(
      await screen.findByRole('option', { name: 'EXL3 2.52bpw (model default)' }),
    ).toBeTruthy();
  });
});

describe('lookup-table picker', () => {
  it('shows only for models with n-gram tables and labels the resolved default', async () => {
    history.replaceState(null, '', '/?m=Qwen/Qwen3.8-27B');
    mount();
    await screen.findByText(/qwen:/);
    await screen.findByTestId('prefill-tps');
    expect(screen.queryAllByRole('combobox', { name: 'Lookup tables' })).toHaveLength(0);
    cleanup();
    history.replaceState(null, '', '/?m=Qwen/Qwen3.8-Flash-Next&lt=nvfp4');
    mount();
    await screen.findByTestId('prefill-tps');
    // The Quantization panel's picker comes before the offload panel's placement picker of the same name.
    const picker = (await screen.findAllByRole('combobox', { name: 'Lookup tables' }))[0];
    expect(picker.textContent).toMatch(/nvfp4/);
    fireEvent.click(picker);
    expect(await screen.findByRole('option', { name: /Model default \(bf16\)/ })).toBeTruthy();
  });
});

describe('estimator inputs', () => {
  it('rejects TP and world-size errors without computing a result', async () => {
    history.replaceState(null, '', '/?m=Qwen/Qwen3.8-27B');
    mount();
    await screen.findByText(/qwen:/);
    await screen.findByTestId('prefill-tps');
    fireEvent.change(screen.getByLabelText(/^TP/), { target: { value: '3' } });
    await waitFor(() => expect(screen.queryByTestId('prefill-tps')).toBeNull());
    expect(screen.getByLabelText(/^TP/).getAttribute('aria-invalid')).toBe('true');
    expect(screen.getByText(/Valid TP:/)).toBeTruthy();
    fireEvent.change(screen.getByLabelText(/^TP/), { target: { value: '1' } });
    expect(screen.getByLabelText(/^DP/)).toBeTruthy();
    expect(screen.queryByLabelText(/^EP/)).toBeNull();
    fireEvent.change(screen.getByLabelText(/^TP/), { target: { value: '2' } });
    expect(screen.getByRole('radiogroup', { name: 'Machines' }).getAttribute('aria-invalid')).toBe(
      'true',
    );
    expect(
      screen.getAllByText(/TP × PP × DP = 2, but the rig has 1 device/).length,
    ).toBeGreaterThan(0);
  });
  it('DP counts devices, validates the batch, enables mode-B EP values, and shows in the rig line', async () => {
    history.replaceState(null, '', '/?m=Qwen/Qwen3.8-Flash-Next&n=4&tp=4');
    mount();
    await screen.findByTestId('prefill-tps');
    expect(screen.getByRole('button', { name: 'About DP' })).toBeTruthy();
    expect(screen.getByText('Valid DP: 1, 2, 3, 4')).toBeTruthy();
    expect(screen.getByText('Valid EP: 1, 2, 4')).toBeTruthy();
    expect(screen.getByRole('heading', { level: 1 }).textContent).not.toMatch(/DP/);
    // DP 2 on top of TP 4: 8 devices for a rig of 4.
    // Exactly "DP": a MoE model also has the "DP attention + EP" layout radio.
    fireEvent.change(screen.getByLabelText(/^DP$/), { target: { value: '2' } });
    await waitFor(() => expect(screen.queryByTestId('prefill-tps')).toBeNull());
    expect(
      screen.getAllByText(/TP × PP × DP = 8, but the rig has 4 devices/).length,
    ).toBeGreaterThan(0);
    fireEvent.change(screen.getByLabelText(/^TP/), { target: { value: '2' } });
    // TP 2 × DP 2: batch 1 cannot split over 2 replicas.
    await waitFor(() =>
      expect(screen.getByLabelText(/^Batch/).getAttribute('aria-invalid')).toBe('true'),
    );
    expect(
      screen.getAllByText('Batch 1 does not split evenly over DP 2. Use a multiple of 2.').length,
    ).toBeGreaterThan(0);
    fireEvent.change(screen.getByLabelText(/^Batch/), { target: { value: '2' } });
    await screen.findByTestId('prefill-tps');
    // EP 4 spans both replicas (a multiple of TP 2 that divides TP × DP = 4).
    expect(screen.getByText('Valid EP: 1, 2, 4')).toBeTruthy();
    expect(screen.getByRole('heading', { level: 1 }).textContent).toMatch(/with TP 2, DP 2$/);
    await waitFor(() => expect(new URLSearchParams(location.search).get('dp')).toBe('2'));
  });
  it('shows EP only for MoE models, with its hint, and blocks the result on an invalid EP', async () => {
    history.replaceState(null, '', '/?m=Qwen/Qwen3.8-Flash-Next&n=2&tp=2&ep=2');
    mount();
    await screen.findByTestId('prefill-tps');
    // /^EP$/: the EP balance fields ("EP imbalance") also start with EP
    const ep = screen.getByLabelText(/^EP$/) as HTMLInputElement;
    expect(ep.value).toBe('2');
    expect(screen.getByRole('button', { name: 'About EP' })).toBeTruthy();
    expect(screen.getByText('Valid EP: 1, 2')).toBeTruthy();
    fireEvent.change(ep, { target: { value: '3' } });
    await waitFor(() => expect(screen.queryByTestId('prefill-tps')).toBeNull());
    expect(screen.getByLabelText(/^EP$/).getAttribute('aria-invalid')).toBe('true');
    expect(screen.getAllByText('EP 3 must divide TP 2.').length).toBeGreaterThan(0);
    expect(screen.getByLabelText(/^TP/).getAttribute('aria-invalid')).not.toBe('true');
    fireEvent.change(screen.getByLabelText(/^EP$/), { target: { value: '2' } });
    await screen.findByTestId('prefill-tps');
    // TP 1 with EP 2: the error is on EP, not TP.
    fireEvent.change(screen.getByLabelText(/^TP/), { target: { value: '1' } });
    await waitFor(() =>
      expect(screen.getByLabelText(/^EP$/).getAttribute('aria-invalid')).toBe('true'),
    );
    expect(screen.getAllByText('EP 2 must divide TP 1.').length).toBeGreaterThan(0);
    expect(screen.getByLabelText(/^TP/).getAttribute('aria-invalid')).not.toBe('true');
  });
  it('hides EP on a dense model and ignores an ep from an old URL', async () => {
    history.replaceState(null, '', '/?m=Qwen/Qwen3.8-27B&n=2&tp=2&ep=2');
    mount();
    await screen.findByTestId('prefill-tps');
    expect(screen.queryByLabelText(/^EP/)).toBeNull();
    expect(screen.queryByRole('button', { name: 'About EP' })).toBeNull();
    expect(screen.queryByText(/Valid EP/)).toBeNull();
  });
  it('restores the URL and writes canonical state after a change', async () => {
    history.replaceState(
      null,
      '',
      '/?m=Qwen/Qwen3.8-27B&d=rtx-4090&u=vram-mod.48gb&n=2&tp=2&kv=fp8_e4m3&b=4',
    );
    mount();
    await screen.findByText(/qwen:/);
    expect(screen.getByRole('combobox', { name: 'Device' }).textContent).toMatch(/RTX 4090/);
    expect(screen.getByRole('combobox', { name: 'KV cache' }).textContent).toMatch(/fp8/i);
    expect((screen.getByLabelText(/^TP/) as HTMLInputElement).value).toBe('2');
    expect((screen.getByLabelText(/^Batch size/) as HTMLInputElement).value).toBe('4');
    fireEvent.change(screen.getByLabelText(/^Batch size/), { target: { value: '8' } });
    await waitFor(() => expect(new URLSearchParams(location.search).get('b')).toBe('8'));
    expect(location.search.slice(1)).toBe(encodeState(decodeState(location.search).state));
    expect(sessionStorage.getItem('specmaxx:last-query')).toBe(location.search);
    history.replaceState(null, '', '/?b=2');
    fireEvent.popState(window);
    await waitFor(() =>
      expect((screen.getByLabelText(/^Batch size/) as HTMLInputElement).value).toBe('2'),
    );
  });
  it('prompts for a gated token, stores it, retries and forgets it without leaking into the URL', async () => {
    let gated = true;
    const loader = vi.fn<ModelLoader>().mockImplementation(async (id, options) => {
      if (id === 'zai-org/GLM-5.3' && gated) {
        gated = false;
        return { kind: 'unauthorized', message: 'Gated model. Add a token.', status: 401 };
      }
      return fixtureLoader(id, options);
    });
    mount(loader);
    fireEvent.change(screen.getByRole('combobox', { name: 'Hugging Face model ID' }), {
      target: { value: 'zai-org/GLM-5.3' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Load' }));
    fireEvent.change(await screen.findByLabelText('Hugging Face token'), {
      target: { value: 'hf_test_secret' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Save token and retry' }));
    await waitFor(() => expect(loader.mock.calls.at(-1)?.[1]?.token).toBe('hf_test_secret'));
    expect(localStorage.getItem(HF_TOKEN_KEY)).toBe('hf_test_secret');
    expect(location.search).not.toContain('secret');
    fireEvent.click(screen.getByRole('button', { name: 'Forget token' }));
    expect(localStorage.getItem(HF_TOKEN_KEY)).toBeNull();
  });
  it('resets quantization overrides to the model defaults', async () => {
    mount();
    fireEvent.click(screen.getByRole('combobox', { name: 'Weights' }));
    fireEvent.click(await screen.findByRole('option', { name: /int4/i }));
    expect(screen.getByRole('combobox', { name: 'Weights' }).textContent).toMatch(/int4/i);
    fireEvent.click(screen.getByRole('button', { name: 'Reset to model default' }));
    expect(screen.getByRole('combobox', { name: 'Weights' }).textContent).not.toMatch(/int4/i);
    expect(screen.queryByRole('button', { name: 'Reset to model default' })).toBeNull();
    await screen.findByTestId('decode-tps');
  });
  it('KV cache: FP4-family formats under "More KV cache formats" show in both pickers; bf16 clears the second', async () => {
    mount();
    const main = () => screen.getByRole('combobox', { name: 'KV cache' });
    const other = () => screen.getByRole('combobox', { name: 'KV cache (other formats)' });
    fireEvent.click(screen.getByText('More KV cache formats'));
    expect(other().textContent).toMatch(/Not used/);
    fireEvent.click(other());
    fireEvent.click(await screen.findByRole('option', { name: 'nvfp4' }));
    expect(main().textContent).toBe('nvfp4');
    expect(other().textContent).toBe('nvfp4');
    await waitFor(() => expect(new URLSearchParams(location.search).get('kv')).toBe('nvfp4'));
    fireEvent.click(main());
    fireEvent.click(await screen.findByRole('option', { name: 'bf16' }));
    expect(main().textContent).toBe('bf16');
    expect(other().textContent).toMatch(/Not used/);
    await waitFor(() => expect(new URLSearchParams(location.search).get('kv')).toBe('bf16'));
  });
  it('a kv= dtype outside the main list (int4) still shows in the main picker', async () => {
    history.replaceState(null, '', '/?kv=int4');
    mount();
    expect(screen.getByRole('combobox', { name: 'KV cache' }).textContent).toBe('int4');
  });
  it('EXL3 weights at 2.5 bpw change the result and survive a URL round-trip', async () => {
    history.replaceState(null, '', '/?m=Qwen/Qwen3.8-27B');
    const first = mount();
    await screen.findByText(/qwen:/);
    const decodeBf16 = (await screen.findByTestId('decode-tps')).textContent;
    expect(screen.queryByLabelText('Bits per weight')).toBeNull();
    fireEvent.click(screen.getByRole('combobox', { name: 'Weights' }));
    fireEvent.click(await screen.findByRole('option', { name: 'EXL3 2.5bpw' }));
    await waitFor(() => expect(new URLSearchParams(location.search).get('bpw')).toBe('2.5'));
    expect(new URLSearchParams(location.search).get('w')).toBe('exl3');
    expect(screen.queryByLabelText('Bits per weight')).toBeNull();
    expect(screen.getByLabelText('Head bits per weight')).toBeTruthy();
    expect(screen.getByRole('button', { name: 'About Head bits per weight' })).toBeTruthy(); // the "?" help icon
    const decode25 = screen.getByTestId('decode-tps').textContent;
    expect(decode25).not.toBe(decodeBf16);

    fireEvent.click(screen.getByRole('combobox', { name: 'Weights' }));
    fireEvent.click(await screen.findByRole('option', { name: 'EXL3 custom…' }));
    const bpw = screen.getByLabelText('Bits per weight') as HTMLInputElement;
    expect(bpw.value).toBe('2.5');
    expect(bpw.step).toBe('0.05');
    fireEvent.change(bpw, { target: { value: '4' } });
    await waitFor(() => expect(screen.getByTestId('decode-tps').textContent).not.toBe(decode25));
    fireEvent.change(bpw, { target: { value: '2.5' } });
    await waitFor(() => expect(screen.getByTestId('decode-tps').textContent).toBe(decode25));

    // Round trip: a fresh app from the written URL selects the 2.5 preset and shows the same result.
    const url = location.search;
    first.unmount();
    history.replaceState(null, '', `/${url}`);
    mount();
    await screen.findByText(/qwen:/);
    await waitFor(() =>
      expect(screen.getByRole('combobox', { name: 'Weights' }).textContent).toMatch(/EXL3 2\.5bpw/),
    );
    expect(screen.queryByLabelText('Bits per weight')).toBeNull();
    await waitFor(() => expect(screen.getByTestId('decode-tps').textContent).toBe(decode25));
  });
  it('ignores stale HF responses', async () => {
    const pending = new Map<string, (value: Awaited<ReturnType<ModelLoader>>) => void>();
    const loader: ModelLoader = (id) =>
      id === 'Qwen/Qwen3.8-27B'
        ? fixtureLoader(id)
        : new Promise((r) => {
            pending.set(id, r);
          });
    mount(loader);
    fireEvent.change(screen.getByRole('combobox', { name: 'Hugging Face model ID' }), {
      target: { value: 'zai-org/GLM-5.3' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Load' }));
    fireEvent.change(screen.getByRole('combobox', { name: 'Hugging Face model ID' }), {
      target: { value: 'Qwen/Qwen3.8-Flash-Next' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Load' }));
    pending.get('Qwen/Qwen3.8-Flash-Next')?.(
      mapHfModel(fixtureRepoFiles('Qwen/Qwen3.8-Flash-Next')),
    );
    await screen.findByTestId('decode-tps');
    pending.get('zai-org/GLM-5.3')?.(mapHfModel(fixtureRepoFiles('zai-org/GLM-5.3')));
    await new Promise((r) => setTimeout(r, 300));
    expect(location.search).toContain('Flash-Next');
    expect(screen.queryByText(/glm/i, { selector: '[data-testid="model-facts"] *' })).toBeNull();
  });
  it('keeps revision and quant overrides when reloading the same model', async () => {
    history.replaceState(null, '', '/?m=Qwen/Qwen3.8-27B&rev=test-revision&w=int4');
    const loader = vi.fn<ModelLoader>().mockImplementation(fixtureLoader);
    mount(loader);
    await screen.findByText(/qwen:/);
    fireEvent.click(screen.getByRole('button', { name: 'Load' }));
    await waitFor(() => expect(loader).toHaveBeenCalledTimes(2));
    expect(loader.mock.calls.at(-1)?.[1]?.revision).toBe('test-revision');
    expect(screen.getByRole('combobox', { name: 'Weights' }).textContent).toMatch(/int4/i);
    await waitFor(() =>
      expect(new URLSearchParams(location.search).get('rev')).toBe('test-revision'),
    );
  });
  it('explains field errors in the results pane without assuming the input layout', async () => {
    history.replaceState(null, '', '/?m=Qwen/Qwen3.8-27B&n=2&tp=3');
    const { container } = mount();
    await screen.findByText(/qwen:/);
    const output = container.querySelector('.output')!;
    expect(output.textContent).toMatch(/TP × PP × DP = 3, but the rig has 2 devices/);
    expect(output.textContent).not.toMatch(/inputs below/);
  });
  it('opens on the default state: Qwen3.8-27B, 8k / 8k, numbers without any loader call', async () => {
    const loader = vi.fn<ModelLoader>().mockRejectedValue(new Error('offline'));
    mount(loader);
    expect((screen.getByLabelText(/^Prefill length/) as HTMLInputElement).value).toBe('8192');
    expect((screen.getByLabelText(/^Decode context/) as HTMLInputElement).value).toBe('8192');
    expect(screen.getByTestId('model-facts').textContent).toMatch(
      /\d+(\.\d+)?B total, \d+(\.\d+)?B active, \d+ layers/,
    );
    expect((await screen.findByTestId('decode-tps')).textContent).toMatch(/[1-9]/);
    expect(screen.queryByRole('alert')).toBeNull();
    await waitFor(() => expect(document.title).toMatch(/@ 8k/));
  });
  it('picking a popular setup sets the counts and links in the URL', async () => {
    mount();
    fireEvent.click(screen.getByRole('combobox', { name: 'Device' }));
    expect(await screen.findByRole('group', { name: 'Popular setups' })).toBeTruthy();
    fireEvent.click(await screen.findByRole('option', { name: /RTX 5090 32GB ×2/ }));
    expect(screen.getByRole('combobox', { name: 'Device' }).textContent).toMatch(
      /RTX 5090 32GB ×2/,
    );
    await waitFor(() => expect(new URLSearchParams(location.search).get('n')).toBe('2'));
    expect(new URLSearchParams(location.search).get('il')).toBeTruthy();
  });
  it('picking the DGX Spark ×4 ring writes fab=ring and names the cabling in the hero', async () => {
    history.replaceState(null, '', '/?m=Qwen/Qwen3.8-27B');
    mount();
    await screen.findByTestId('prefill-tps');
    fireEvent.click(screen.getByRole('combobox', { name: 'Device' }));
    fireEvent.click(await screen.findByRole('option', { name: /DGX Spark.*×4 · ConnectX-7 ring/ }));
    await waitFor(() => expect(new URLSearchParams(location.search).get('fab')).toBe('ring'));
    expect(screen.getByRole('combobox', { name: 'Cabling' }).textContent).toBe('Ring');
    await waitFor(() =>
      expect(document.querySelector('.rig')?.textContent).toMatch(
        /on 4 × NVIDIA DGX Spark \(GB10, 128GB memory\) over ConnectX-7 \(ring\), with/,
      ),
    );
    // Changing the cabling leaves the setup: the trigger shows the device again.
    fireEvent.click(screen.getByRole('combobox', { name: 'Cabling' }));
    fireEvent.click(await screen.findByRole('option', { name: 'Switch' }));
    await waitFor(() => expect(new URLSearchParams(location.search).get('fab')).toBeNull());
    expect(screen.getByRole('combobox', { name: 'Device' }).textContent).toMatch(
      /×4 · ConnectX-7 switch/,
    );
    expect(document.querySelector('.rig')?.textContent).not.toMatch(/\(ring\)/);
  });
  it('offers only the cablings the ports allow, and one as plain text', async () => {
    history.replaceState(null, '', '/?d=ryzen-ai-max-plus-395&nodes=3');
    mount();
    await screen.findByTestId('setup-row');
    fireEvent.click(screen.getByRole('combobox', { name: 'Cabling' }));
    const names = (await screen.findAllByRole('option')).map((o) =>
      o.textContent?.replace('✓', ''),
    );
    expect(names).toEqual(['Full mesh', 'Daisy chain']);
    fireEvent.keyDown(screen.getByRole('combobox', { name: 'Cabling' }), { key: 'Escape' });
    cleanup();
    history.replaceState(null, '', '/?d=dgx-spark&nodes=2');
    mount();
    await screen.findByTestId('setup-row');
    expect(screen.queryByRole('combobox', { name: 'Cabling' })).toBeTruthy();
    cleanup();
    history.replaceState(null, '', '/?n=2&nodes=2');
    mount();
    await screen.findByTestId('setup-row');
    expect(screen.queryByRole('combobox', { name: 'Cabling' })).toBeNull();
    expect(screen.getByTestId('cabling').textContent).toBe('Switch');
  });
  it('offers the 2-card NVLink bridge only at 2 GPUs, and more GPUs fall back to PCIe', async () => {
    history.replaceState(null, '', '/?d=rtx-3090&n=2&il=rtx-3090-nvlink-bridge');
    mount();
    await screen.findByTestId('setup-row');
    fireEvent.click(screen.getByRole('combobox', { name: 'Local link' }));
    expect(
      (await screen.findAllByRole('option')).some((o) => /NVLink bridge/.test(o.textContent ?? '')),
    ).toBe(true);
    fireEvent.keyDown(screen.getByRole('combobox', { name: 'Local link' }), { key: 'Escape' });
    const perMachine = screen.getByRole('radiogroup', { name: 'GPUs in one machine' });
    fireEvent.click(within(perMachine).getByRole('radio', { name: '4' }));
    await waitFor(() => expect(new URLSearchParams(location.search).get('il')).toBeNull());
    expect(new URLSearchParams(location.search).get('n')).toBe('4');
    fireEvent.click(screen.getByRole('combobox', { name: 'Local link' }));
    const options = await screen.findAllByRole('option');
    expect(options.length).toBeGreaterThan(0);
    expect(options.some((o) => /NVLink bridge/.test(o.textContent ?? ''))).toBe(false);
  });
  it.each([
    ['×4 · ConnectX-7 switch', ''],
    ['×4 · USB4 ring', '?n=2&tp=2'],
    ['×4 · ConnectX-7 ring', '?n=3&pp=3'],
  ])('picking %s from %j chooses TP and PP that use all 4 devices', async (name, query) => {
    history.replaceState(null, '', `/${query}${query ? '&' : '?'}m=Qwen/Qwen3.8-27B`);
    mount();
    await screen.findByTestId('prefill-tps');
    fireEvent.click(screen.getByRole('combobox', { name: 'Device' }));
    fireEvent.click(await screen.findByRole('option', { name: new RegExp(name) }));
    await waitFor(() => expect(screen.getByTestId('device-total').textContent).toMatch(/= 4 /));
    const tp = Number((screen.getByLabelText(/^TP/) as HTMLInputElement).value);
    const pp = Number((screen.getByLabelText(/^PP/) as HTMLInputElement).value);
    expect(tp * pp).toBe(4);
    expect(await screen.findByTestId('prefill-tps')).toBeTruthy();
    expect(screen.queryByText(/unused|must be equal/)).toBeNull();
  });
  it('a huge device count shows validation errors instead of hanging', async () => {
    history.replaceState(null, '', '/?m=Qwen/Qwen3.8-27B&n=1024');
    mount();
    await waitFor(() => expect(screen.queryByTestId('prefill-tps')).toBeNull());
    expect(screen.getAllByText(/must be equal|TP × PP/).length).toBeGreaterThan(0);
  });
  it('TP × PP below the device count blocks the simulation', async () => {
    history.replaceState(null, '', '/?m=Qwen/Qwen3.8-27B&n=4&tp=2');
    mount();
    await screen.findByText(/qwen:/);
    expect(
      await screen.findAllByText(/TP × PP × DP = 2, but the rig has 4 devices/),
    ).not.toHaveLength(0);
    expect(screen.queryByTestId('prefill-tps')).toBeNull();
  });
  it('filters the device list by search and selects with the keyboard', async () => {
    mount();
    fireEvent.click(screen.getByRole('combobox', { name: 'Device' }));
    const search = await screen.findByRole('combobox', { name: /search/i });
    fireEvent.change(search, { target: { value: '4090' } });
    const options = screen.getAllByRole('option');
    expect(options.length).toBeGreaterThan(0);
    expect(options.every((o) => /4090/.test(o.textContent ?? ''))).toBe(true);
    fireEvent.keyDown(search, { key: 'ArrowDown' });
    fireEvent.keyDown(search, { key: 'Enter' });
    await waitFor(() =>
      expect(screen.getByRole('combobox', { name: 'Device' }).textContent).toMatch(/4090/),
    );
  });
  it('shows the prefetch accuracy only with offload, 80 by default, and writes opf', async () => {
    history.replaceState(null, '', '/?m=Qwen/Qwen3.8-Flash-Next');
    mount();
    await screen.findByTestId('decode-tps');
    expect(screen.queryByLabelText('Prefetch accuracy %')).toBeNull();
    cleanup();
    history.replaceState(null, '', '/?m=Qwen/Qwen3.8-Flash-Next&ox=ram');
    mount();
    await screen.findByTestId('decode-tps');
    const field = screen.getByLabelText('Prefetch accuracy %') as HTMLInputElement;
    expect(field.value).toBe('80');
    fireEvent.change(field, { target: { value: '90' } });
    await waitFor(() => expect(location.search).toContain('opf=0.9'));
    fireEvent.change(screen.getByLabelText('Prefetch accuracy %'), { target: { value: '80' } });
    await waitFor(() => expect(location.search).not.toContain('opf='));
  });
  it('splits routed experts between host RAM and SSD: half by default, then the count and the end', async () => {
    history.replaceState(null, '', '/?m=Qwen/Qwen3.8-Flash-Next&olt=ssd&ox=ram');
    mount();
    await screen.findByTestId('decode-tps');
    expect(screen.queryByLabelText('Layers in host RAM')).toBeNull();
    fireEvent.click(screen.getByRole('combobox', { name: 'MoE experts' }));
    fireEvent.click(await screen.findByRole('option', { name: 'Host RAM + SSD' }));
    await waitFor(() => expect(location.search).toContain('ox=split&oxr=24'));
    const ram = screen.getByLabelText('Layers in host RAM') as HTMLInputElement;
    expect(ram.value).toBe('24');
    fireEvent.change(ram, { target: { value: '10' } });
    await waitFor(() => expect(location.search).toContain('oxr=10'));
    fireEvent.click(screen.getByRole('combobox', { name: 'Host RAM holds' }));
    fireEvent.click(await screen.findByRole('option', { name: 'Latest layers' }));
    await waitFor(() => expect(location.search).toContain('oxr=10&oxe=last'));
    // Back to one tier: the split keys go.
    fireEvent.click(screen.getByRole('combobox', { name: 'MoE experts' }));
    fireEvent.click(await screen.findByRole('option', { name: 'SSD' }));
    await waitFor(() => expect(location.search).toContain('ox=ssd'));
    expect(location.search).not.toMatch(/ox[re]=/);
  });
  it('offers no lookup-table split for a model with a single table layer (Qwen PLE)', async () => {
    history.replaceState(null, '', '/?m=Qwen/Qwen3.8-Flash-Next&olt=ssd');
    mount();
    await screen.findByTestId('decode-tps');
    const section = screen.getByTestId('offload-advanced');
    fireEvent.click(within(section).getByRole('combobox', { name: 'Lookup tables' }));
    expect(await screen.findByRole('option', { name: 'SSD' })).toBeTruthy();
    expect(screen.queryByRole('option', { name: 'Host RAM + SSD' })).toBeNull();
  });
  it('splits lookup tables between host RAM and SSD by table layer, in Advanced', async () => {
    history.replaceState(null, '', '/?m=deepseek-ai/DeepSeek-V4.1-Flash&olt=ssd');
    mount();
    await screen.findByTestId('decode-tps');
    const section = screen.getByTestId('offload-advanced');
    const picker = within(section).getByRole('combobox', { name: 'Lookup tables' });
    fireEvent.click(picker);
    fireEvent.click(await screen.findByRole('option', { name: 'Host RAM + SSD' }));
    await waitFor(() => expect(location.search).toMatch(/olt=split&olr=1/));
    const ram = within(section).getByLabelText('Layers in host RAM') as HTMLInputElement;
    fireEvent.change(ram, { target: { value: '2' } });
    await waitFor(() => expect(location.search).toContain('olt=split&olr=2'));
    fireEvent.click(within(section).getByRole('combobox', { name: 'Host RAM holds' }));
    fireEvent.click(await screen.findByRole('option', { name: 'Latest layers' }));
    await waitFor(() => expect(location.search).toContain('olr=2&ole=last'));
    // With a split, the scale layout defaults to each tier's own (no `ols`).
    expect(within(section).getByRole('combobox', { name: 'Scale layout' }).textContent).toContain(
      'Each tier’s default',
    );
    expect(location.search).not.toContain('ols=');
  });
  it('every scale layout offered on a lookup-table split still gives a result', async () => {
    history.replaceState(null, '', '/?m=deepseek-ai/DeepSeek-V4.1-Flash&olt=split');
    mount();
    await screen.findByTestId('decode-tps');
    const section = screen.getByTestId('offload-advanced');
    const layout = () => within(section).getByRole('combobox', { name: 'Scale layout' });
    const names = [
      'Each tier’s default',
      'Separate scale reads',
      'Scales pinned in host RAM',
      'Interleaved rows',
    ];
    fireEvent.click(layout());
    expect(
      (await screen.findAllByRole('option')).map((o) => o.textContent?.replace('✓', '')),
    ).toEqual(names);
    fireEvent.click(screen.getByRole('option', { name: names[0] }));
    for (const name of [...names.slice(1), names[0]]) {
      fireEvent.click(layout());
      fireEvent.click(await screen.findByRole('option', { name }));
      await waitFor(() => expect(layout().textContent).toContain(name));
      // The engine accepts the rules: the result stays, and no validation error shows.
      await waitFor(() => expect(screen.getByTestId('decode-tps').textContent).toMatch(/[1-9]/));
      expect(screen.queryByText(/pinned_host/)).toBeNull();
    }
  });
  it('per-op costs in Advanced: the latency floor writes lf, the dispatch overlay ed=1 and its hero figure', async () => {
    mount();
    await screen.findByTestId('decode-tps');
    const section = screen.getByTestId('per-op-costs-advanced');
    const floor = within(section).getByLabelText('Per-op latency floor µs') as HTMLInputElement;
    expect(floor.value).toBe('0');
    fireEvent.change(floor, { target: { value: '1' } });
    await waitFor(() => expect(location.search).toContain('lf=1'));
    fireEvent.change(floor, { target: { value: '0' } });
    await waitFor(() => expect(location.search).not.toContain('lf='));
    expect(screen.queryByTestId('decode-dispatch')).toBeNull();
    fireEvent.click(within(section).getByLabelText('on'));
    await waitFor(() => expect(location.search).toContain('ed=1'));
    await screen.findByTestId('decode-dispatch');
    fireEvent.click(within(section).getByLabelText('off'));
    await waitFor(() => expect(location.search).not.toContain('ed='));
  });
  it(
    'expert all-to-all in Advanced: only in MoE mode B; writes dsp, cmb and a2e',
    async () => {
      history.replaceState(null, '', '/?m=zai-org/GLM-5.3-Flash&d=gb200&n=72&b=72&dp=72');
      mount();
      await screen.findByTestId('decode-tps');
      expect(screen.queryByTestId('all-to-all-advanced')).toBeNull();
      cleanup();
      history.replaceState(null, '', '/?m=zai-org/GLM-5.3-Flash&d=gb200&n=72&b=72&dp=72&ep=72');
      mount();
      await screen.findByTestId('decode-tps');
      const section = screen.getByTestId('all-to-all-advanced');
      const eff = within(section).getByLabelText('All-to-all efficiency %') as HTMLInputElement;
      expect(eff.value).toBe('100');
      fireEvent.change(eff, { target: { value: '70' } });
      await waitFor(() => expect(location.search).toContain('a2e=70'));
      fireEvent.click(within(section).getByRole('combobox', { name: 'Dispatch format' }));
      fireEvent.click(await screen.findByRole('option', { name: 'NVFP4' }));
      await waitFor(() => expect(location.search).toContain('dsp=nvfp4'));
      fireEvent.click(within(section).getByRole('combobox', { name: 'Combine format' }));
      fireEvent.click(await screen.findByRole('option', { name: 'FP8' }));
      await waitFor(() => expect(location.search).toContain('cmb=fp8_e4m3'));
      await screen.findByTestId('decode-tps');
      fireEvent.change(eff, { target: { value: '100' } });
      await waitFor(() => expect(location.search).not.toContain('a2e='));
    },
    HEAVY_APP_TEST_TIMEOUT_MS,
  );
  it(
    'network in Advanced: NICs for 2 × 8 H100 write nic; in-network reduction writes inr; hidden on one RTX 5090',
    async () => {
      mount();
      await screen.findByTestId('decode-tps');
      expect(screen.queryByTestId('network-advanced')).toBeNull();
      cleanup();
      history.replaceState(
        null,
        '',
        '/?m=zai-org/GLM-5.3-Flash&d=h100-sxm&n=8&nodes=2&tp=16&ov=none',
      );
      mount();
      await screen.findByTestId('decode-tps');
      const section = screen.getByTestId('network-advanced');
      const nics = within(section).getByRole('combobox', { name: 'NICs per machine' });
      expect(nics.textContent).toContain('8 (1 per GPU)');
      // Without overlap the prefill all-reduces between the machines show: half the NIC speed per GPU slows them.
      const prefillBefore = screen.getByTestId('prefill-tps').textContent;
      fireEvent.click(nics);
      fireEvent.click(await screen.findByRole('option', { name: '4 (1 per 2 GPUs)' }));
      await waitFor(() => expect(location.search).toContain('nic=4'));
      await waitFor(() =>
        expect(document.querySelector('.rig')?.textContent).toContain('(4 NICs per machine)'),
      );
      await waitFor(() =>
        expect(screen.getByTestId('prefill-tps').textContent).not.toBe(prefillBefore),
      );
      const inr = within(section).getByRole('combobox', { name: 'In-network reduction' });
      expect(inr.textContent).toContain('Off');
      fireEvent.click(inr);
      fireEvent.click(await screen.findByRole('option', { name: 'On (NVLS, SHARP)' }));
      await waitFor(() => expect(location.search).toContain('inr=on'));
      // Back to one NIC per GPU: the default, nothing in the URL.
      fireEvent.click(nics);
      fireEvent.click(await screen.findByRole('option', { name: '8 (1 per GPU)' }));
      await waitFor(() => expect(location.search).not.toContain('nic='));
    },
    HEAVY_APP_TEST_TIMEOUT_MS,
  );
  it(
    'communication overlap in Advanced: rows for the classes the run has; writes ove and tbo=1',
    async () => {
      mount();
      await screen.findByTestId('decode-tps');
      // One device: no communication, no fieldset.
      expect(screen.queryByTestId('overlap-advanced')).toBeNull();
      cleanup();
      history.replaceState(null, '', '/?m=zai-org/GLM-5.3-Flash&d=gb200&n=72&b=72&dp=72&ep=72');
      mount();
      await screen.findByTestId('decode-tps');
      const section = screen.getByTestId('overlap-advanced');
      expect(within(section).queryByRole('combobox', { name: 'All-reduce (TP)' })).toBeNull();
      expect(within(section).queryByRole('combobox', { name: 'Stage hand-off (PP)' })).toBeNull();
      const ep = within(section).getByRole('combobox', { name: 'All-to-all (EP)' });
      expect(ep.textContent).toContain('Default (full)');
      fireEvent.click(ep);
      fireEvent.click(await screen.findByRole('option', { name: 'Partial' }));
      await waitFor(() => expect(location.search).toContain('ove=50'));
      const percent = within(section).getByLabelText('All-to-all hidden %') as HTMLInputElement;
      expect(percent.value).toBe('50');
      // Typing 0 keeps the percent field open (still Partial while editing).
      fireEvent.change(percent, { target: { value: '0' } });
      await waitFor(() => expect(location.search).toContain('ove=0'));
      expect(within(section).getByLabelText('All-to-all hidden %')).toBeTruthy();
      fireEvent.change(percent, { target: { value: '25' } });
      await waitFor(() => expect(location.search).toContain('ove=25'));
      // Two-batch overlap on clears the explicit EP choice: EP goes to its two-batch default, full.
      fireEvent.click(within(section).getByLabelText('on'));
      await waitFor(() => expect(location.search).toContain('tbo=1'));
      expect(location.search).not.toContain('ove=');
      expect(ep.textContent).toContain('Default (full)');
      expect(within(section).queryByLabelText('All-to-all hidden %')).toBeNull();
      fireEvent.click(ep);
      fireEvent.click(await screen.findByRole('option', { name: 'None' }));
      await waitFor(() => expect(location.search).toContain('ove=0'));
      await screen.findByTestId('decode-tps');
      fireEvent.click(within(section).getByLabelText('off'));
      await waitFor(() => expect(location.search).not.toContain('tbo='));
      fireEvent.click(ep);
      fireEvent.click(await screen.findByRole('option', { name: 'Default (full)' }));
      await waitFor(() => expect(location.search).not.toContain('ove='));
    },
    HEAVY_APP_TEST_TIMEOUT_MS,
  );
  it(
    'DP-only lookup sharing exposes the existing TP overlap setting and persists ovt',
    async () => {
      history.replaceState(null, '', '/?m=Qwen/Qwen3.8-Flash-Next&n=2&dp=2&b=2&ltdp=1&ovt=0');
      mount();
      await screen.findByTestId('decode-tps');
      const section = screen.getByTestId('overlap-advanced');
      const picker = within(section).getByRole('combobox', { name: 'TP / lookup exchange' });
      expect(picker.textContent).toContain('None');
      fireEvent.click(picker);
      fireEvent.click(await screen.findByRole('option', { name: 'Full' }));
      await waitFor(() => expect(location.search).toContain('ovt=100'));
      fireEvent.click(screen.getByRole('combobox', { name: 'Share lookup tables across DP' }));
      fireEvent.click(await screen.findByRole('option', { name: 'off' }));
      await waitFor(() => expect(screen.queryByTestId('overlap-advanced')).toBeNull());
      await waitFor(() => expect(location.search).not.toContain('ltdp'));
      expect(location.search).toContain('ovt=100');
    },
    HEAVY_APP_TEST_TIMEOUT_MS,
  );

  it(
    'TBO applicability with shared tables follows effective prefix KV placement, not its saved choice',
    async () => {
      history.replaceState(
        null,
        '',
        '/?m=Qwen/Qwen3.8-Flash-Next&n=2&dp=2&ep=2&b=4&ltdp=1&okt=ram&okp=25',
      );
      mount();
      await screen.findByTestId('decode-tps');
      expect(screen.getByRole('radiogroup', { name: 'Two-batch overlap' })).toBeTruthy();
      fireEvent.change(screen.getByRole('spinbutton', { name: 'Prefix cache hit %' }), {
        target: { value: '50' },
      });
      await waitFor(() =>
        expect(screen.queryByRole('radiogroup', { name: 'Two-batch overlap' })).toBeNull(),
      );
    },
    HEAVY_APP_TEST_TIMEOUT_MS,
  );

  it('keeps Advanced closed until opened, then its subsections are open', async () => {
    const { container } = mount();
    await screen.findByTestId('decode-tps');
    // The first Advanced is the Hardware panel's (it comes before the Parallelism panel's).
    const advanced = container.querySelector('details.advanced') as HTMLDetailsElement;
    expect(advanced.contains(screen.getByTestId('offload-advanced'))).toBe(true);
    expect(advanced.open).toBe(false);
    advanced.open = true;
    expect((container.querySelector('details.spec-editor') as HTMLDetailsElement).open).toBe(true);
  });
  it.each([
    {
      name: 'offload off',
      query: '?m=Qwen/Qwen3.8-27B',
      offload: ['Token embeddings', 'KV cache on GPU %'],
    },
    {
      name: 'an old whole-layer offload URL (ignored)',
      query: '?m=Qwen/Qwen3.8-27B&odl=32',
      offload: ['Token embeddings', 'KV cache on GPU %'],
    },
    {
      name: 'a lookup-table model at DP 1',
      query: '?m=Qwen/Qwen3.8-Flash-Next',
      offload: ['Token embeddings', 'Lookup tables', 'KV cache on GPU %'],
    },
    {
      name: 'expert offload on a lookup-table model',
      query: '?m=Qwen/Qwen3.8-Flash-Next&n=2&dp=2&b=2&ox=ram',
      offload: [
        'Token embeddings',
        'Lookup tables',
        'Share lookup tables across DP',
        'KV cache on GPU %',
        'Queue depth',
        'Prefetch accuracy %',
      ],
    },
    {
      name: 'a lookup-table split',
      query: '?m=deepseek-ai/DeepSeek-V4.1-Flash&n=2&dp=2&b=2&olt=split',
      offload: [
        'Token embeddings',
        'Lookup tables',
        'Share lookup tables across DP',
        'Layers in host RAM',
        'Host RAM holds',
        'Scale layout',
        'Queue depth',
        'SSD read rate',
        'Prefetch accuracy %',
      ],
    },
  ])(
    'shows one Advanced in Hardware, with an Offload subsection for every model ($name)',
    async ({ query, offload }) => {
      history.replaceState(null, '', `/${query}`);
      const { container } = mount();
      await screen.findByTestId('decode-tps');
      const panel = (title: string) =>
        [...container.querySelectorAll('details.panel')].find(
          (p) => p.querySelector(':scope > summary strong')?.textContent === title,
        )!;
      const advancedOf = (title: string) =>
        [...panel(title).querySelectorAll('summary')].filter((s) => s.textContent === 'Advanced');
      const summaries = advancedOf('Hardware');
      expect(summaries).toHaveLength(1);
      // The Parallelism panel has its own Advanced (DCP), and no other panel has one.
      expect(advancedOf('Parallelism')).toHaveLength(1);
      expect(
        [...container.querySelectorAll('summary')].filter((s) => s.textContent === 'Advanced'),
      ).toHaveLength(2);
      expect(
        within(advancedOf('Parallelism')[0].parentElement!).getByTestId('parallel-advanced'),
      ).toBeTruthy();
      const advanced = summaries[0].parentElement as HTMLDetailsElement;
      expect(advanced.open).toBe(false);
      expect(within(screen.getByTestId('offload-section')).queryByText('Advanced')).toBeNull();
      const section = screen.queryByTestId('offload-advanced');
      expect(section).not.toBeNull();
      expect(advanced.contains(section)).toBe(true);
      // The subsection comes first, before the device unlocks and the spec editor.
      expect(section!.compareDocumentPosition(advanced.querySelector('details.spec-editor')!)).toBe(
        Node.DOCUMENT_POSITION_FOLLOWING,
      );
      expect(section!.querySelector('legend')!.textContent).toBe('Offload');
      const labels = [...section!.querySelectorAll('.field-label')].map((l) =>
        l.textContent?.trim(),
      );
      expect(labels).toEqual(offload);
    },
  );
  it('table sharing is available only for Device, and changing the tier clears the saved flag', async () => {
    history.replaceState(null, '', '/?m=deepseek-ai/DeepSeek-V4.1-Flash&n=2&dp=2&b=2&ltdp=1');
    mount();
    await screen.findByTestId('decode-tps');
    const sharing = () =>
      screen.getByRole('combobox', { name: 'Share lookup tables across DP' }) as HTMLButtonElement;
    expect(sharing().disabled).toBe(false);
    expect(sharing().textContent).toContain('on');
    const placement = () =>
      within(screen.getByTestId('offload-advanced')).getByRole('combobox', {
        name: 'Lookup tables',
      });
    for (const tier of ['Host RAM', 'SSD', 'Host RAM + SSD']) {
      fireEvent.click(placement());
      fireEvent.click(await screen.findByRole('option', { name: tier }));
      expect(sharing().disabled).toBe(true);
      expect(sharing().textContent).toContain('off');
      await waitFor(() => expect(location.search).not.toContain('ltdp'));
    }
    fireEvent.click(placement());
    fireEvent.click(await screen.findByRole('option', { name: 'Device' }));
    expect(sharing().disabled).toBe(false);
    expect(sharing().textContent).toContain('off');
    fireEvent.click(sharing());
    fireEvent.click(await screen.findByRole('option', { name: 'on' }));
    await waitFor(() => expect(location.search).toContain('ltdp=1'));
  });

  it('token embeddings: Host RAM writes oet=ram and brings in the host controls; On GPU removes it', async () => {
    history.replaceState(null, '', '/?m=Qwen/Qwen3.8-27B');
    mount();
    await screen.findByTestId('decode-tps');
    const picker = () =>
      within(screen.getByTestId('offload-advanced')).getByRole('combobox', {
        name: 'Token embeddings',
      });
    expect(picker().textContent).toBe('On GPU');
    expect(screen.queryByRole('combobox', { name: 'Host RAM' })).toBeNull();
    fireEvent.click(picker());
    expect(
      (await screen.findAllByRole('option')).map((o) => o.textContent?.replace('✓', '')),
    ).toEqual(['On GPU', 'Host RAM', 'SSD']);
    fireEvent.click(screen.getByRole('option', { name: 'Host RAM' }));
    await waitFor(() => expect(location.search).toContain('oet=ram'));
    expect(screen.getByRole('combobox', { name: 'Host RAM' })).toBeTruthy();
    fireEvent.click(picker());
    fireEvent.click(await screen.findByRole('option', { name: 'On GPU' }));
    await waitFor(() => expect(location.search).not.toContain('oet='));
    expect(screen.queryByRole('combobox', { name: 'Host RAM' })).toBeNull();
  });
  it('token embeddings on unified memory: In memory and SSD only; a shared oet=ram shows In memory', async () => {
    history.replaceState(null, '', '/?m=Qwen/Qwen3.8-27B&d=apple-m3-ultra-80c');
    mount();
    await screen.findByTestId('decode-tps');
    const picker = () =>
      within(screen.getByTestId('offload-advanced')).getByRole('combobox', {
        name: 'Token embeddings',
      });
    expect(picker().textContent).toBe('In memory');
    fireEvent.click(picker());
    expect(
      (await screen.findAllByRole('option')).map((o) => o.textContent?.replace('✓', '')),
    ).toEqual(['In memory', 'SSD']);
    cleanup();
    history.replaceState(null, '', '/?m=Qwen/Qwen3.8-27B&d=apple-m3-ultra-80c&oet=ram');
    mount();
    await screen.findByTestId('decode-tps');
    expect(picker().textContent).toBe('In memory');
  });
  it('token embeddings of a tied model: a disabled picker whose help says why, and an oet in the URL is ignored', async () => {
    history.replaceState(null, '', '/?m=Qwen/Qwen3.8-27B&oet=ssd');
    mount(async (id) => {
      const r = await fixtureLoader(id);
      return r.kind === 'ok' ? { ...r, model: { ...r.model, tieEmbeddings: true } } : r;
    });
    await screen.findByTestId('decode-tps');
    const trigger = within(screen.getByTestId('offload-advanced')).getByRole('combobox', {
      name: 'Token embeddings',
    }) as HTMLButtonElement;
    expect(trigger.disabled).toBe(true);
    expect(trigger.textContent).toBe('On GPU');
    // Nothing is placed: no SSD or host controls, no notice.
    expect(screen.queryByRole('combobox', { name: 'SSD' })).toBeNull();
    expect(screen.queryByRole('alert')).toBeNull();
    fireEvent.focus(screen.getByRole('button', { name: 'About Token embeddings' }));
    expect(
      await screen.findByText(
        'This model reuses this table for its output, so it stays on the GPU.',
      ),
    ).toBeTruthy();
    expect(screen.getByTestId('decode-tps').textContent).toMatch(/[1-9]/);
  });
  it('labels the counts in plain language with a total', async () => {
    history.replaceState(null, '', '/?n=4');
    mount();
    expect(screen.getByRole('radiogroup', { name: 'GPUs in one machine' })).toBeTruthy();
    expect(
      (
        within(screen.getByRole('radiogroup', { name: 'GPUs in one machine' })).getByRole('radio', {
          name: '4',
        }) as HTMLInputElement
      ).checked,
    ).toBe(true);
    expect(screen.getByRole('radiogroup', { name: 'Machines' })).toBeTruthy();
    expect(screen.getByTestId('device-total').textContent).toMatch(/= 4 GPUs total/);
  });
  it('typing a model ID and pressing Enter loads it', async () => {
    const loader = vi.fn<ModelLoader>().mockImplementation(fixtureLoader);
    mount(loader);
    const input = screen.getByRole('combobox', { name: 'Hugging Face model ID' });
    fireEvent.change(input, { target: { value: 'zai-org/GLM-5.3-Flash' } });
    fireEvent.keyDown(input, { key: 'Enter' });
    await waitFor(() =>
      expect(loader.mock.calls.some((c) => c[0] === 'zai-org/GLM-5.3-Flash')).toBe(true),
    );
  });
});

describe('config read from another repo', () => {
  /**
   * Mentions of where a config was read from; the user wants none of them shown. `unsloth/` is the mirror's org as
   * a repo path (a hardware note names "Unsloth" the trainer, which is unrelated).
   */
  const FROM_TEXT = /read from|unsloth\//i;
  const official = 'meta-models/Muse-Spark-Open-816B-A42B';
  const mirror = 'unsloth/Muse-Spark-Open-816B-A42B';

  /** The Hub without a token: the official repo answers 401, the mirror serves its checked-in fixtures. */
  const gatedHub = (async (url: string) => {
    if (url.includes(official)) return new Response('', { status: 401 });
    const kind =
      url === apiUrl(mirror) ? 'api' : url === resolveUrl(mirror, 'config.json') ? 'config' : null;
    const body = kind && readFixtureJson(`${fixtureStem(mirror)}.${kind}.json`);
    return body ? new Response(JSON.stringify(body)) : new Response('', { status: 404 });
  }) as unknown as typeof fetch;
  /** A stand-in muse_spark family (a dense 8B shape) used only until the real family is registered. */
  const stub: FamilyEntry = {
    family: 'muse_spark',
    modelTypes: ['muse_spark', 'muse_spark_text'],
    architectures: ['MuseSparkForConditionalGeneration'],
    build: (_text, _top, files) => ({
      kind: 'ok',
      model: { ...REF_DENSE_8B, name: files.id, source: files.id },
      notes: [],
      mtpParamsEstimate: 0,
      visionParamsEstimate: 0,
      excludedParts: [],
    }),
  };
  const families = FAMILIES.some((f) => f.family === 'muse_spark') ? FAMILIES : [...FAMILIES, stub];
  /** `loadHfModel` with the fake Hub (and the stand-in family while the real one is missing). */
  const gatedLoader: ModelLoader = async (id, options) => {
    const files = await fetchHfRepoFiles(id, {
      ...options,
      fetchImpl: gatedHub,
      cache: createCache(memoryStorage(), new Map()),
      tokenStorage: memoryStorage(),
    });
    return isHfError(files) ? files : mapHfModel(files, families);
  };

  it('loads a gated model from its mirror without a token prompt or any mention of the mirror', async () => {
    history.replaceState(null, '', `/?m=${official}`);
    mount(gatedLoader);
    await waitFor(() =>
      expect(screen.getByTestId('model-facts').textContent).toMatch(/muse_spark/),
    );
    expect((await screen.findByTestId('decode-tps')).textContent).toMatch(/[1-9]/);
    expect(screen.queryByLabelText('Hugging Face token')).toBeNull();
    expect(screen.queryByRole('alert')).toBeNull();
    expect(document.body.textContent).not.toMatch(FROM_TEXT);
  });

  it('does not show the base repo a Mistral NVFP4 config was read from', async () => {
    const id = 'mistralai/Mistral-Large-4-1T-A52B-NVFP4';
    history.replaceState(null, '', `/?m=${id}`);
    mount(async (requested) =>
      requested === id
        ? mapHfModel(
            fixtureRepoFiles(id, { dir: 'quant', configFrom: 'mistralai/Mistral-Large-4-1T-A52B' }),
          )
        : fixtureLoader(requested),
    );
    await waitFor(() => expect(screen.getByTestId('model-facts').textContent).toMatch(/mistral/i));
    await screen.findByTestId('decode-tps');
    expect(document.body.textContent).not.toMatch(/read from/i);
  });
});

describe('speculative-decoding acceptance hints', () => {
  it('flags a preset α used at a k it was not fitted at, and clears the flag at a fitted k or a user α', async () => {
    history.replaceState(null, '', '/?m=Qwen/Qwen3.8-27B&sd=dflash2&sdr=hyp');
    mount();
    await screen.findByText(/qwen:/);
    const section = await screen.findByTestId('specdec-section');
    const k = within(section).getByLabelText('Draft length k');
    // DFlash 2 prose 0.79 is fitted at k 7 (Qwen3.8 card) only; the k 15 reports are cited, not fitted. Default k 7.
    expect((k as HTMLInputElement).value).toBe('7');
    expect(within(section).queryByText(/fitted at k/)).toBeNull();
    fireEvent.change(k, { target: { value: '5' } });
    expect(await within(section).findByText('⚠ fitted at k=7')).toBeTruthy();
    fireEvent.change(k, { target: { value: '7' } });
    await waitFor(() => expect(within(section).queryByText(/fitted at k/)).toBeNull());
    // A user α is not a preset: no warning at any k.
    fireEvent.change(k, { target: { value: '5' } });
    fireEvent.change(within(section).getByLabelText('Acceptance α'), { target: { value: '0.8' } });
    await waitFor(() => expect(within(section).queryByText(/fitted at k/)).toBeNull());
  });
  it('labels DFlash 2 agentic as DFlash v1 data fitted at k 15', async () => {
    history.replaceState(null, '', '/?m=Qwen/Qwen3.8-27B&sd=dflash2&sdr=hyp&sdu=agentic');
    mount();
    await screen.findByText(/qwen:/);
    const section = await screen.findByTestId('specdec-section');
    expect(await within(section).findByText('from DFlash v1 · ⚠ fitted at k=15')).toBeTruthy();
  });
  it('labels a pooled preset with the k of its runs, with ⚠ at any other k', async () => {
    history.replaceState(null, '', '/?m=Qwen/Qwen3.8-27B&sd=dspark&sdr=hyp&sdu=code');
    mount();
    await screen.findByText(/qwen:/);
    const section = await screen.findByTestId('specdec-section');
    // default k 7 is one of the pooled runs' k (paper k 7, Inco k 15)
    expect(await within(section).findByText('pooled from k=7, 15')).toBeTruthy();
    fireEvent.change(within(section).getByLabelText('Draft length k'), { target: { value: '5' } });
    expect(await within(section).findByText('⚠ pooled from k=7, 15')).toBeTruthy();
  });
});

describe('drafter picker', () => {
  it('offers the checkpoint’s own drafter as the model default and returns to it from Hypothetical', async () => {
    history.replaceState(null, '', '/?m=deepseek-ai/DeepSeek-V4.1-Flash&sd=dspark');
    mount();
    const section = await screen.findByTestId('specdec-section');
    const drafter = within(section).getByRole('combobox', { name: 'Drafter' }) as HTMLInputElement;
    const defaultLabel = 'Model default (checkpoint’s DSpark drafter)';
    expect(drafter.value).toBe(defaultLabel);

    fireEvent.focus(drafter);
    fireEvent.click(await screen.findByRole('option', { name: /Hypothetical drafter/ }));
    await waitFor(() => expect(new URLSearchParams(location.search).get('sdr')).toBe('hyp'));
    expect(drafter.value).toBe('Hypothetical drafter');

    fireEvent.focus(drafter);
    const options = await screen.findAllByRole('option');
    expect(options.map((o) => o.textContent)).toEqual([
      expect.stringContaining(defaultLabel),
      expect.stringContaining('Hypothetical drafter'),
    ]);
    fireEvent.click(options[0]);
    await waitFor(() => expect(new URLSearchParams(location.search).has('sdr')).toBe(false));
    expect(drafter.value).toBe(defaultLabel);
    expect(within(section).queryByTestId('hypothetical-badge')).toBeNull();
  });
});

describe('drafter picker keyboard and custom repos', () => {
  it('highlights the model default with the keyboard, and lists everything after a custom repo is committed', async () => {
    history.replaceState(null, '', '/?m=deepseek-ai/DeepSeek-V4.1-Flash&sd=dspark&sdr=hyp');
    mount();
    const section = await screen.findByTestId('specdec-section');
    const drafter = within(section).getByRole('combobox', { name: 'Drafter' }) as HTMLInputElement;
    fireEvent.focus(drafter);
    await screen.findAllByRole('option');
    fireEvent.keyDown(drafter, { key: 'ArrowDown' });
    const first = screen.getAllByRole('option')[0];
    expect(first.textContent).toContain('Model default');
    await waitFor(() => expect(first.getAttribute('aria-selected')).toBe('true'));
    fireEvent.keyDown(drafter, { key: 'Enter' });
    await waitFor(() => expect(new URLSearchParams(location.search).has('sdr')).toBe(false));

    fireEvent.change(drafter, { target: { value: 'someone/my-drafter' } });
    fireEvent.keyDown(drafter, { key: 'Enter' });
    await waitFor(() =>
      expect(new URLSearchParams(location.search).get('sdr')).toBe('someone/my-drafter'),
    );
    expect(drafter.value).toBe('someone/my-drafter');
    fireEvent.focus(drafter);
    expect((await screen.findAllByRole('option')).map((o) => o.textContent)).toEqual([
      expect.stringContaining('Model default'),
      expect.stringContaining('Hypothetical drafter'),
    ]);
  });
});

describe('prefill/decode disaggregation', () => {
  const QUERY =
    '?m=zai-org/GLM-5.3-Flash&d=h100-sxm&n=8&nodes=4&dp=32&ep=32&lay=dep&b=32&pd=1&pdr=1:3' +
    '&dec.d=h100-sxm&dec.n=8&dec.nodes=18&dec.dp=144&dec.ep=144&dec.lay=dep&dec.b=144';

  it('shows both pools, the System answer with its quote, and edits the decode pool through the pool switch', async () => {
    history.replaceState(null, '', `/${QUERY}`);
    mount();
    const system = await screen.findByTestId('pd-system-tps', {}, { timeout: 5000 });
    expect(system.textContent).toMatch(/tok\/s\/GPU @ 8k\/1k, decode @ 8k$/);
    expect(screen.getByTestId('pd-balanced').textContent).toMatch(/^(1 : [\d.]+|[\d.]+ : 1)$/);
    expect(screen.getByTestId('pd-kv').textContent).toMatch(/per request @ 8k/);
    expect(screen.getByTestId('pd-rig').textContent).toContain('DP attention + EP 32 / 144');
    await waitFor(() =>
      expect(document.title).toMatch(/tok\/s @ 8k prefill · .* tok\/s @ 8k decode/),
    );
    // The per-pool panels name the edited pool; the decode pool's batch goes to dec.b.
    expect(screen.getByText('Hardware (prefill pool)')).toBeTruthy();
    const pool = screen.getByRole('radiogroup', { name: 'Edit pool' });
    fireEvent.click(within(pool).getByRole('radio', { name: 'Decode' }));
    expect(screen.getByText('Hardware (decode pool)')).toBeTruthy();
    const batch = screen.getByRole('spinbutton', { name: 'Batch size (decode pool)' });
    expect((batch as HTMLInputElement).value).toBe('144');
    fireEvent.change(batch, { target: { value: '288' } });
    await waitFor(() => expect(location.search).toContain('dec.b=288'));
    expect(location.search).toContain('&b=32&');
  });

  it.each([
    [1, 2],
    [2, 1],
    [1, 1],
  ])(
    'lookup sharing control considers both pools: prefill DP %i / decode DP %i',
    async (prefillDp, decodeDp) => {
      history.replaceState(
        null,
        '',
        `/?m=Qwen/Qwen3.8-Flash-Next&n=${prefillDp}&dp=${prefillDp}&b=2&ltdp=1&pd=1&dec.n=${decodeDp}&dec.dp=${decodeDp}&dec.b=2`,
      );
      mount();
      await screen.findByTestId('pd-system-tps', {}, { timeout: 5000 });
      const pool = screen.getByRole('radiogroup', { name: 'Edit pool' });
      const sharing = () =>
        screen.queryByRole('combobox', { name: 'Share lookup tables across DP' });
      const applicable = prefillDp > 1 || decodeDp > 1;
      for (const name of ['Prefill', 'Decode']) {
        fireEvent.click(within(pool).getByRole('radio', { name }));
        if (applicable) expect(sharing()?.textContent).toContain('on');
        else expect(sharing()).toBeNull();
      }
      if (applicable) {
        // This is a shared deployment choice even when edited from the DP1 pool.
        fireEvent.click(
          within(pool).getByRole('radio', { name: prefillDp === 1 ? 'Prefill' : 'Decode' }),
        );
        fireEvent.click(sharing()!);
        fireEvent.click(await screen.findByRole('option', { name: 'off' }));
        await waitFor(() => expect(location.search).not.toContain('ltdp'));
        fireEvent.click(
          within(pool).getByRole('radio', { name: prefillDp === 1 ? 'Decode' : 'Prefill' }),
        );
        expect(sharing()?.textContent).toContain('off');
      }
    },
    HEAVY_APP_TEST_TIMEOUT_MS,
  );

  it('turning it off gives back the single-pool page and URL', async () => {
    const query = '?m=zai-org/GLM-5.3-Flash&d=h100-sxm&n=8&nodes=4&dp=32&ep=32&lay=dep&b=32';
    history.replaceState(null, '', `/${query}`);
    mount();
    await waitFor(() => expect(screen.getByTestId('decode-tps').textContent).toMatch(/[1-9]/));
    const before = screen.getByTestId('decode-tps').textContent;
    const toggle = screen.getByRole('radiogroup', { name: 'Prefill/decode disaggregation' });
    fireEvent.click(within(toggle).getByRole('radio', { name: 'On' }));
    await screen.findByTestId('pd-system-tps', {}, { timeout: 5000 });
    await waitFor(() => expect(location.search).toContain('pd=1'));
    fireEvent.click(within(toggle).getByRole('radio', { name: 'Off' }));
    await waitFor(() => expect(screen.queryByTestId('pd-system-tps')).toBeNull());
    await waitFor(() => expect(screen.getByTestId('decode-tps').textContent).toBe(before));
    await waitFor(() => expect(location.search).toBe(query));
  });
});
