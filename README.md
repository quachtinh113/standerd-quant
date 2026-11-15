# Standard Quant

Hệ thống giao dịch có hệ thống theo kiến trúc 5 khối của Robert Carver (*Systematic Trading*, 2015).
Tái sử dụng các nguyên tắc thể chế của `Word_Quant/05_Quant` nhưng gọn, thuần Python (numpy + pandas), có test.

```
Data Rules → Forecast Generation → Volatility Targeting → Portfolio Construction → Execution
 s1_data_rules   s2_forecast         s3_vol_targeting       s4_portfolio            s5_execution
```

Luồng dữ liệu một chiều: khối N chỉ import khối ≤ N. Kiểu dữ liệu giữa các khối ở `standard_quant/contracts.py`.

| Khối | Việc làm | Công thức / quy tắc chính | Mã nguồn |
|---|---|---|---|
| 1. Data Rules | Làm sạch giá, ước lượng vol | Chỉ ffill (không bfill), loại tick lỗi; vol = 70% EWMA35 + 30% trung bình 10 năm | `s1_data_rules/` |
| 2. Forecast | Quy tắc → forecast chuẩn hoá | EWMAC (8/16/32/64 × 4), Breakout; \|trung bình\| ≈ 10, kẹp ±20; trộn + FDM | `s2_forecast/` |
| 3. Vol Targeting | Forecast → lot | `lot = vốn × σ_mục_tiêu/√256 / (vol giá × point_value) × forecast/10`; buffer 10% | `s3_vol_targeting/` |
| 4. Portfolio | Gộp instrument | Trọng số (equal / inverse-vol), IDM = 1/√(wᵀCw), kẹp [1, 2.5] | `s4_portfolio/` |
| 5. Execution | Target − hiện tại → lệnh | Drawdown Ladder, SL vật lý, cấm DCA, halt chỉ giảm | `s5_execution/` |

## Chạy

```bash
python -m pytest tests -q          # 78 test (~35 giây)
python run_alpha_report.py         # đánh giá toàn bộ alpha trong registry
python run_meta_report.py          # L3: meta-labeling có giúp thật không? (AUC purged-CV, DSR, PBO)
python run_backtest.py             # dữ liệu tổng hợp (chỉ minh hoạ)
python run_backtest.py --csv XAUUSD=gold.csv EURUSD=eur.csv   # cột time, close (bất kỳ khung, tự gộp D1)
```

## Tầng L2 — Alpha (Finding Alphas, Tulchinsky)

`standard_quant/s2_forecast/alpha/` nằm trong khối 2: alpha chỉ là một nguồn forecast nữa, đi tiếp qua vol targeting như EWMAC.

| File | Vai trò |
|---|---|
| `operators.py` | Time-series: `ts_delta/mean/std/zscore/rank/corr/cov/decay_linear/ema…`; cross-sectional: `cs_rank/demean/zscore/scale/winsorize`. Tất cả nhân quả; `ts_delay(d<0)` bị từ chối |
| `library.py` | 5 alpha mẫu (`@register`), mỗi cái 1 ý tưởng kinh tế, ít tham số |
| `bridge.py` | Điểm số thô → forecast (chia \|alpha\| expanding ×10, kẹp ±20) |
| `evaluate.py` | Sharpe, turnover, fitness = SR·√(\|ret\|/max(turnover, 0.125)), IC, tách IS/OOS, ngưỡng Sharpe theo số lần thử (Bailey–López de Prado), robustness (bỏ từng instrument, trễ +1, chi phí ×2) |

Dùng alpha trong hệ thống: thêm `"alphas": ["mom_vol_adj", "range_position"]` vào config. Thêm alpha mới:

```python
@register("my_alpha")
def my_alpha(d):                       # d["close"], d["returns"], d["vol"] là DataFrame rộng
    return -ops.ts_zscore(d["returns"], 20)
```

Quy trình bắt buộc trước khi bật alpha vào config: `evaluate_alpha(..., n_trials=<số alpha/biến thể đã thử>)` phải `passes=True` **và** `robustness_report` không sụp đổ. `n_trials` phải khai báo trung thực, nếu không ngưỡng chống overfit vô nghĩa.

Giới hạn: universe hiện chỉ 3 instrument nên toán tử cắt ngang (`cs_*`) rất nhiễu; chúng chỉ có ý nghĩa khi mở rộng lên hàng chục instrument.

## Tầng L3 — Meta-Labeling (AFML, López de Prado)

`standard_quant/s2_forecast/meta/`. L2 quyết định **hướng**; ML chỉ trả lời "lần vào lệnh này có thắng không?" và chỉ được **thu nhỏ** forecast về [0, 1]×, không bao giờ đảo chiều hay tạo lệnh.

| File | AFML | Vai trò |
|---|---|---|
| `labeling.py` | ch.3–4 | Triple-barrier (±mult×vol×√h, rào dọc h phiên) → nhãn thắng/thua; trọng số uniqueness cho nhãn chồng lấn |
| `features.py` | ch.5 | 6 đặc trưng nhân quả (\|forecast\|, vol_ratio, mom5/20, rank60, trail_hit chỉ từ nhãn đã phân giải `t1 < t`) |
| `cv.py` | ch.7 | Purged K-Fold + embargo |
| `model.py` | ch.6, 10 | RandomForest nông (depth 3, leaf ≥ 40, balanced); walk-forward refit mỗi 63 phiên chỉ trên nhãn `t1 < T`; `bet_size` gate/continuous |
| `apply.py` | | `apply_meta`, `evaluate_meta` (AUC OOS, hit-rate giữ/bỏ, Sharpe có/không meta) |
| `validation.py` | ch.11–15 | PSR, DSR, PBO (CSCV) — cổng chống overfit cho backtest |

Bật trong config: `"meta_enabled": true` (+ `meta_threshold`, `meta_mode`, `meta_burn_in`, `meta_min_events`). Giai đoạn chưa đủ nhãn: `pass` (giữ nguyên forecast) hoặc `flat` (về 0).

**Điều kiện bật trên tiền thật** (bắt buộc, theo thứ tự): (1) AUC Purged-CV > ~0.53 ổn định theo thời gian; (2) uplift OOS walk-forward dương sau chi phí; (3) DSR > 0.95 với `n_trials` khai báo gồm cả số ngưỡng, đặc trưng, tham số barrier đã thử; (4) PBO < 0.5 trên lưới cấu hình đủ rộng. `python run_meta_report.py` in cả bốn.

**Lưu ý đọc kết quả**: trên dữ liệu tổng hợp, AUC Purged-CV ≈ 0.50 (không có thông tin) trong khi Sharpe proxy "có meta" cao hơn rõ rệt ở vài ngưỡng. Đó là dấu hiệu của chọn ngưỡng/biến động mẫu chứ không phải alpha: AUC mới là thước đo chính, uplift Sharpe chỉ đáng tin khi AUC đã vượt 0.5.

Cần `scikit-learn` (đã kiểm với 1.7.2).

## L5 — Execution Isolation (kiểu Citadel)

`standard_quant/l5_execution/` là một **tiến trình riêng** chỉ làm ba việc: đọc tín hiệu → bắn lệnh → ghi latency.

```
[L1–L4: pipeline]  --publish_intents()-->  signals.db (SQLite/WAL)  --claim_next()-->  [L5 worker] --> MT5
   (tính toán)        s5_execution/publisher.py      điểm giao duy nhất               store/executor/gateway
```

| File | Vai trò |
|---|---|
| `store.py` | Hộp thư tín hiệu. Trạng thái `PENDING→CLAIMED→SENT/REJECTED/FAILED/UNKNOWN`, `EXPIRED` theo TTL. Claim nguyên tử, `signal_id` duy nhất (idempotent) |
| `executor.py` | Kiểm tra bất biến (từ chối, không sửa) → gửi → ghi `latency`: queue_ms, roundtrip_ms, total_ms, trượt giá bps, retcode, ticket. `latency_summary()` cho p50/p95/p99 |
| `gateway.py` | `MT5Gateway` (mỏng, import lười, filling mode động, đóng đúng ticket trên tài khoản hedging) và `FakeGateway` cho test |
| `__main__.py` | `python -m standard_quant.l5_execution --db signals.db --symbols XAUUSDm --gateway mt5 [--summary]` |

**Điều L5 KHÔNG được làm (test canh giữ, `tests/test_l5_isolation.py`):**
- chỉ import thư viện chuẩn + `MetaTrader5`; cấm numpy/pandas/sklearn/scipy/talib và mọi module `standard_quant.*` khác (AST whitelist);
- import `standard_quant.l5_execution` không được nạp numpy/pandas/sklearn (kiểm bằng subprocess; `standard_quant/__init__` nạp lười);
- không có từ vựng chỉ báo (`ewm`, `rolling`, `atr`, `forecast`, `kelly`, `predict`…) trong mã L5.

**An toàn khi gửi lệnh (đều có test):** allowlist symbol bắt buộc (rỗng = từ chối hết) · giới hạn lot/lệnh · lệnh mở phải có SL đúng phía · `halt`/`reduce_only` flag (chỉ cho đóng khi halt) · TTL (tín hiệu cũ không bao giờ được bắn) · chặn nếu giá live lệch `ref_price` quá `max_slippage_bps` · retry **chỉ** với requote/price-changed · timeout/ngoại lệ → `UNKNOWN`, **không bao giờ tự gửi lại** · khởi động lại: mọi `CLAIMED` mồ côi → `UNKNOWN` · hai worker không thể cùng claim một tín hiệu.

**Chưa kiểm chứng:** `MT5Gateway` chưa chạy trên MT5/tài khoản thật (máy dev không có terminal) — cần chạy demo account và so khớp retcode, filling mode, đóng vị thế trước khi dùng. Latency `ms_queue` dùng đồng hồ tường (cùng máy), `ms_roundtrip` dùng `perf_counter`. Đối soát lệnh `UNKNOWN` với `positions_get` hiện làm thủ công.

## Bot hoàn chỉnh: L4 Rủi ro · L6 Giám sát · Orchestrator

```
CSV/MT5 → [L1 data → L2 alpha → L3 meta → vol target → portfolio] → intents
        → [L4 RiskDesk: duyệt trước giao dịch + Drawdown Ladder + sổ băm] → signals.db
        → [L5 worker, tiến trình riêng] → MT5 → latency/TCA → [L6 health/alerts/report]
```

| Tầng | Mã | Việc làm |
|---|---|---|
| Orchestrator | `bot.py`, `run_bot.py`, `io_feeds.py` | Một chu kỳ/nến D1 đóng: nạp dữ liệu → pipeline → L4 → hộp thư. Idempotent theo nến (`last_asof`), từ chối dữ liệu cũ (`max_stale_days`), **không phát tín hiệu cho symbol đang có tín hiệu chưa chốt/UNKNOWN** (chống nhân đôi vị thế) |
| L4 Risk | `l4_risk/desk.py` | Cổng duyệt cho mọi intent mở/tăng: halt (ladder/file thủ công), thứ 6 ≥ 21:00/cuối tuần, rollover 23:45–00:05, SL bắt buộc, trần lot/symbol, số vị thế tối đa, margin level, đòn bẩy cộng dồn. **Lệnh giảm/đóng luôn qua.** Mọi từ chối có lý do + ghi sổ. Đẩy cờ `halt` xuống L5; ladder (HWM, latch) lưu bền qua khởi động lại |
| Sổ kiểm toán | `ledger.py` | JSONL chuỗi băm SHA-256 (seq, prev_record_hash, record_hash), phát hiện sửa/xoá/đứt chuỗi. Bố cục trường như `audit_risk_ledger.jsonl` của 05_Quant; **thuật toán băm riêng, không tương thích byte với vibe_ledger** |
| L6 Monitoring | `l6_monitoring/` | `check_health`: worker L5 im lặng (heartbeat), lệnh UNKNOWN, tín hiệu tồn đọng, tỉ lệ từ chối, p95 latency, trượt giá TB, cờ halt, sổ bị sửa. `position_drift`, `unknown_for_review` + `resolve_unknown` (chỉ người quyết định), `Dispatcher` chống spam, báo cáo JSON ghi nguyên tử. CLI trả mã thoát 0/1/2 cho Task Scheduler |

Chạy thật (3 tiến trình, mặc định không gửi lệnh nếu không bật worker):
```bash
python run_bot.py --db data/signals.db --feed mt5 --map XAUUSD=XAUUSDm --magic 260910 --loop      # quyết định
python -m standard_quant.l5_execution --db data/signals.db --symbols XAUUSDm --gateway mt5 --magic 260910   # thực thi
python -m standard_quant.l6_monitoring --db data/signals.db --ledger data/signals.ledger.jsonl --out data/report.json  # giám sát
```
Lưu ý: `--map` đổi tên nội bộ → symbol broker ở **bot**, còn `--symbols` của L5 phải là **symbol broker** (vd `XAUUSDm`); `Instrument` trong config khoá theo tên nội bộ.

**Chưa kiểm chứng trên môi trường thật:** `MT5Feed`, `MT5Account`, `MT5Gateway` (không có terminal MT5 ở máy dev). Trước khi dùng tiền: chạy demo account ≥ 2 tuần, so khớp vị thế bot vs MT5 (`position_drift`), kiểm tra filling mode/retcode, mô phỏng crash giữa chừng để xem đường `UNKNOWN`.

## Bất biến (có test)

1. **Zero look-ahead**: forecast/vol/xác suất meta tại t không đổi khi cắt dữ liệu sau t (`test_*_causal`, `test_pipeline_no_lookahead_end_to_end`). Khớp lệnh ở phiên kế (`exec_lag ≥ 1`).
2. **Drawdown Ladder** từ HWM: −3% cảnh báo, −5% ×0.5, −7.5% halt (latch tới khi người duyệt); lỗ ngày ≥ cap chỉ halt trong ngày.
3. **Không Kelly đầy đủ**: kích thước = vol target (mặc định 20%/năm).
4. **Mọi lệnh mở/tăng có SL** phía broker (`giá ∓ 6 × vol`, disaster stop).
5. **Cấm martingale/DCA**: `max_dca_layers = 0` (Config từ chối khác 0); không tăng vị thế đang lỗ.

## Ánh xạ từ 05_Quant

| Standard Quant | Nguồn trong 05_Quant |
|---|---|
| `s5_execution/risk.py` | `All_Fleet_Bot/core/fleet_risk_guard.py` (ladder) |
| `s3` + `s4` | `quant_research_hub/03_alpha_factory/execution_cost_and_slippage_model.py` (`CarverVolatilityTargetingSizer`) |
| `s1` làm sạch | `quant_data_lake/`, nguyên tắc point-in-time |
| `s2` rules | thay cho logic SMA/Donchian rời rạc của `strategies/bot*.py` |

## Chưa làm (cố ý)

- **Adapter MT5**: cài `Broker` Protocol (`s5_execution/brokers.py`); `PaperBroker` là tham chiếu. Chưa có kết nối live.
- **Kiểm tra SL trong backtest**: dữ liệu chỉ có close; SL chỉ được sinh ra, không mô phỏng chạm.
- **Carry, handcrafting đầy đủ, ADV cap, chi phí Almgren-Chriss**: chỗ cắm là `s2/rules.py`, `s4/construct.py`, `Instrument.cost_per_lot_rt`.
- Kết quả backtest tổng hợp **không** là bằng chứng alpha. Cần chạy trên dữ liệu thật qua cổng validation (DSR/PBO) của 05_Quant trước khi dùng vốn.
