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
python -m pytest tests -q          # 38 test (~30 giây)
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
