# Moshi KD 데이터셋 계획 — PyTorch Dataset 규격

**범위**: `training/data/` 레이어 설계 (Dataset / Collator / Sampler / Config).
**비범위**: 학습 루프, 손실 함수 구현, 모델 코드.

기존 `data_pipeline/`은 **오프라인 준비(raw → Arrow)** 담당이고, 이 문서는 그 산출물을
`torch.utils.data` 표준 계약으로 소비하는 레이어를 정의한다. `data_pipeline`은 참고 대상이지,
따라야 할 규격이 아니다. 규격은 `Dataset` / `Sampler` / `collate_fn` 3-피스 계약이다.

---

## 0. 전제 (검증된 사실)

| 항목 | 값 | 출처 |
|---|---|---|
| Mimi 상수 | 12.5Hz, K=8, card=2048, 24kHz | [schema.py:27-30](../../data_pipeline/schema.py#L27-L30) |
| 준비 데이터 경로 | `data/prepared/{source}/{split}/` (HF `save_to_disk`) | [prepare_dataset.py:49](../../data_pipeline/prepare_dataset.py#L49) |
| group | `en_kd` / `en_solo` / `ko_tts` / `text_anchor` | [base.py:20](../../data_pipeline/datasets/base.py#L20) |
| split | `train` / `probe` — probe는 **절대 학습 금지** | [prepare_dataset.py:12](../../data_pipeline/prepare_dataset.py#L12) |
| 저장 레이아웃 | **delay-free, frame-aligned**. delay는 collator가 적용 | [en_kd_dataset.py:281](../../data_pipeline/datasets/en_kd_dataset.py#L281) |
| A/B 대칭 | "누가 user인지" 미저장. 학습 시점 결정 → 2× 데이터 | [schema.py:12](../../data_pipeline/schema.py#L12) |
| teacher logits | raw top-k. **temperature는 loss 하이퍼파라미터** | [schema.py:14](../../data_pipeline/schema.py#L14) |
| KD 대상 | codebook 0 (semantic)만. 저장은 K=8 전체(상위집합) | ARCHITECTURE §5 |
| T 범위 | 250 ~ 1500 frames (20s ~ 120s) | `FilterConfig.min_frames` / `GenConfig.max_frames` |

**함정 2개 (설계에 직접 영향)**

1. `source ≠ sample_type`. `en_solo`는 `source="en_solo"`이지만 `sample_type="ko_tts"`
   ([en_solo_dataset.py:57-59](../../data_pipeline/datasets/en_solo_dataset.py#L57-L59)).
   → **동작 분기는 `sample_type`(모양 계약), 믹싱/로깅은 `source`(디렉토리)**. 서로 추론 금지.
2. `text_anchor`는 `codes_a=(8,0)`에 프레임 정렬이 안 된 길이 L 텍스트. 오디오 배치와 섞을 수 없다.

---

## 1. 모듈 레이아웃

```
training/data/
├── item.py       # KDSample TypedDict — 계약의 단일 정의. I/O 없음
├── dataset.py    # MoshiKDDataset: Arrow → KDSample
├── crop.py       # Window / choose_window / apply_window (순수 함수)
├── collator.py   # KDCollator: delay 적용, 패딩, loss weight 텐서
├── text_collator.py  # TextAnchorCollator (별도 경로, §5.4)
├── sampler.py    # GroupIndex, MixingBatchSampler
├── schedule.py   # MixSchedule (한국어 비율 램프)
├── loader.py     # build_dataloader() 팩토리
└── config.py     # dataclass + from_yaml
```

책임 경계는 아래 한 줄로 요약된다.

> `__getitem__`은 **인덱스만의 순수 함수**. 배치나 모델 하이퍼파라미터가 필요한 모든 것은 collator.

---

## 2. Dataset 레이어

### 2.1 클래스

소스별 서브클래싱은 하지 않는다. 4개 소스가 동일 Arrow 스키마를 공유하므로 다형성 이득이 없고
crop/swap 로직만 4곳으로 흩어진다. **단일 클래스 + `sample_type` 키 정규화 함수 dict**.

```python
class MoshiKDDataset(Dataset):
    """하나의 (source, split) → KDSample. 비율/믹싱은 모른다."""
    def __init__(self, root, source, split="train", *, cfg, double_ab=None, seed=0): ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> KDSample: ...
    def set_epoch(self, epoch: int) -> None: ...

_NORMALIZERS = {"en_kd": _normalize_dual,
                "ko_tts": _normalize_solo,       # en_solo 행도 여기로
                "text_anchor": _normalize_text_only}
```

### 2.2 아이템 계약 `KDSample`

전부 **CPU torch 텐서, 언배치, delay-free, 언패딩**. `K=8`.

```python
class KDSample(TypedDict):
    sample_uid: str; source: str; sample_type: str; lang: str
    swapped: Tensor        # bool ()
    is_text_only: Tensor   # bool ()
    num_frames: Tensor     # int32 ()  == T (text_anchor은 0)
    codes_self:  Tensor    # int16 (8, T)
    codes_other: Tensor    # int16 (8, T)  없으면 silence code로 채움
    text_self:   Tensor    # int32 (T,)
    text_other:  Tensor    # int32 (T,)   없으면 PAD
    text_flat:   Tensor    # int32 (L,)   text_anchor 전용, 그 외 (0,)
    has_teacher: Tensor    # bool ()
    topk:        Tensor    # int32 ()
    teacher_val: Tensor    # fp16  (8, T, topk)
    teacher_idx: Tensor    # int16 (8, T, topk)
    use_kd / use_ce_audio / use_ce_text: Tensor   # bool ()
```

`codes_a/b`가 아니라 **`codes_self`/`codes_other`**로 명명한다. `__getitem__`에서 방향이 이미
확정되었으므로 a/b 이름을 남기면 하류에서 재스왑하는 버그를 부른다. `swapped`는 로깅/재현용.

### 2.3 A/B 방향 스왑 — **인덱스 배가(index-doubling), Dataset에서**

```python
def __len__(self):  return self._n_rows * (2 if self.double_ab else 1)
def _resolve(self, i): return (i // 2, bool(i % 2)) if self.double_ab else (i, False)
```

collator에서 랜덤 스왑하지 않는 이유:

- **재현성**: collator RNG는 `num_workers`·prefetch·배치 구성에 결합된다. 워커 수를 4→8로 바꾸면
  같은 시드에서도 데이터 스트림이 달라진다. 인덱스 배가는 워커 수와 무관하다.
- **에폭 의미**: "2× 데이터"가 목표인데 랜덤 스왑은 한 대화를 A-self로 두 번 보고 B-self로
  0번 볼 수 있다. 인덱스 배가는 **모든 대화를 양방향 정확히 1회씩** 보장한다.
- **믹싱 비율**: `mixing_sampler`가 group 단위로 뽑으므로 `len()`이 2배 축소되면 설정 비율이
  조용히 틀어진다.

`double_ab`는 B 스트림이 있는 `en_kd`에서만 유효 — 나머지는 `__init__`에서 강제 `False`.

**스왑 시 teacher는 self 쪽만 유지한다.**
```python
lo, hi = ("b", "a") if swapped else ("a", "b")
teacher_val = arrs[f"teacher_topk_val_{lo}"]   # hi쪽은 버림
```
KD 손실은 모델링 대상 화자의 codebook-0 logits에 걸린다. 반대편은 입력이지 예측 타깃이 아니다.
이 한 줄이 T=1500 기준 샘플당 top-k 전송량을 1.5MB → 0.75MB로 줄인다.
**스왑 후에도 A쪽 teacher를 들고 가는 버그는 크래시 없이 조용히 모델만 나빠진다** → 전용 테스트 필수.

### 2.4 소스별 정규화

- **en_kd**: 그대로. `text_tokens_a.shape[0] == T` assert (프레임 정렬이 구성상 보장되므로
  불일치는 손상된 행 → 조용한 절단 대신 큰 소리로 실패).
- **ko_tts / en_solo**: `codes_b`가 `None`. **silence code로 채우는 것은 collator가 아니라
  Dataset이 한다** (schema.py:16의 주석을 갱신할 것). 배치 결합이 전혀 없는 샘플별 연산이고,
  collator에서 하면 stack 중 `sample_type` 분기가 필요해지며 아이템 스키마의 균일성이 깨진다.
- **text_anchor**: **T=0 퇴화 스트림**. 프레임 필드는 전부 진짜로 빈 텐서, 토큰은 `text_flat`.
  ~~L 프레임짜리 가짜 무음 오디오 생성~~은 기각 — 12.5Hz 타이밍을 날조하고 오디오 CE를 오염시킨다.

### 2.5 silence code — 추측 금지

`cfg.silence_codes: (8,) int16`은 config 주입이며 미설정 시 `__post_init__`에서 raise.
**0으로 기본값을 두면 안 된다. 0은 유효한 Mimi 코드다.**

도출 절차 (오프라인 1회, `configs/data/mimi_silence.json`에 커밋):
1. 24kHz 디지털 무음 20s + −60dBFS 룸톤 20s 합성
2. 동일 체크포인트(`kyutai/moshiko-pytorch-bf16`)로 인코딩
3. 앞뒤 10프레임 폐기 (인코더 수용장 워밍업)
4. codebook별 최빈값 → `silence_codes[k]`, 최빈 비율 > 0.9 assert
5. `mimi_ckpt_id`와 함께 저장, collator가 코퍼스 빌드 id와 일치 검증

**틀렸을 때의 실패 모드** (양성 노이즈가 아니라 *최대로 학습 가능한* 신호):
공유 self/user 임베딩 테이블이 실제로는 등장하지 않는 코드에 거대한 그래디언트를 받고,
Role Token이 역할 태그가 아니라 "user 채널이 합성인가?" 탐지기로 퇴화하며,
user-stream CE가 모든 ko_tts에서 자명하게 풀려 손실은 건강해 보이는데 아무것도 학습되지 않는다.
그리고 이 단서는 `lang=ko`와 완벽히 상관되므로 RISKS §1의 언어↔과제 지름길에 직접 먹인다.

### 2.6 크롭 — Dataset에서, teacher 실체화 **이전**

collator에서 크롭하면 워커가 1500프레임 top-k를 IPC로 직렬화한 뒤 80%를 버린다.
크롭을 먼저 하면 샘플당 1.5MB → ~0.3MB.

- `cfg.max_frames = 750` (60s). 생성 상한 1500은 K=8 soft target과 함께 메모리상 비현실적.
- train은 `random`, **probe는 `center` 강제** (config 신뢰 말고 `__init__`에서 하드 강제).
  probe가 확률적이면 모델과 무관한 이유로 지표가 흔들린다.
- 최소 길이 필터는 여기서 하지 않는다 (`FilterConfig.min_frames=250`이 상류에서 처리).
- delay(≤2프레임)는 크롭 후 collator가 창 내부에서 적용 → 경계 손실은 750 중 2프레임, 무시 가능.
  **Dataset에서 delay만큼 미리 깎지 말 것** (§1의 경계를 넘는다).
- 슬라이스는 view이므로 `np.ascontiguousarray` 명시 호출 — 안 하면 부모 버퍼를 물고 있어
  최적화가 무효화된다.

### 2.7 Arrow 지연 접근 / 워커 안전성

```python
def _hf(self):
    pid = os.getpid()
    if self._hf_ds is None or self._hf_pid != pid:
        self._hf_ds = load_from_disk(str(self.path)); self._hf_pid = pid
    return self._hf_ds
```
PID 키 재오픈으로 `spawn`·`fork` 양쪽에서 정확. `__getstate__`로 Arrow 테이블은 피클에서 제외.

**핫패스에서 `hf_ds[i]`와 `row_to_arrays`를 호출하지 않는다.** 둘 다 전체 컬럼을
(양쪽 teacher top-k 포함) 파이썬 레벨로 실체화한다. 대신 pyarrow 버퍼 zero-copy view →
reshape → **크롭 슬라이스 → 그 다음** `ascontiguousarray`.

```python
flat = col.chunk(...)[...].values.to_numpy(zero_copy_only=True)
v = flat.reshape(K, T, topk)[:, w.start:w.end, :]
```
`zero_copy_only=True`를 명시해 스키마 변경 시 조용한 전체 복사 복귀 대신 즉시 실패하게 한다.
`row_to_arrays`는 **테스트의 정답 오라클**로 남긴다 (fast path와 elementwise 일치 검증).

`hf_ds.set_format("torch")` 금지 — 같은 실체화 경로를 타서 위 최적화를 무효화한다.

### 2.8 dtype / temperature

- `teacher_val` → **fp16**. bf16 모델의 logits이므로 fp16은 teacher 자체 정밀도 대비 무손실이고,
  지배적 텐서를 절반으로 줄인다. 디버그 모드에서 `isfinite` assert (inf는 상류 덤프 손상 신호,
  안 잡으면 KD 손실이 조용히 NaN).
- `teacher_idx` → **int16 유지**. `int64` 변환은 loss의 gather 호출 지점에서. Dataset에서 하면
  워커 IPC 경계를 4배로 넘긴다.
- **temperature 적용 금지.** ① 스윕 대상 하이퍼파라미터라 굽으면 스윕마다 재읽기,
  ② top-k *부분집합*에 대한 `softmax(logits/τ)`는 진짜 tempered 분포가 아니며 꼬리 질량 처리는
  loss의 설계 결정, ③ fp16에서 작은 τ로 나누면 log-sum-exp 전 오버플로 위험.
  `gen_meta.gen_temperature`는 *샘플링* 온도로 KD τ와 무관 — 혼동 방지 주석 필수.

### 2.9 결정성

```python
def _rng_for(self, index):
    return np.random.default_rng(
        np.random.SeedSequence(entropy=self._seed, spawn_key=(self._epoch, index)))
```
`self.rng` 상태 변이 금지 — 워커별 복사본에서 뽑기 순서가 어느 워커가 어느 인덱스를 받았는지에
의존해 재현 불가. `SeedSequence(spawn_key=)`는 워커 수·배치 크기·호출 순서와 무관하게
샘플 `i`·에폭 `e`에 동일한 크롭을 준다.

**`persistent_workers=True`의 함정**: 워커가 첫 fork 시점의 `_epoch`를 들고 있어 매 에폭 동일한
크롭이 나온다. 손실 곡선에 안 보이는 실패다. → **`persistent_workers=False` + 에폭마다
`set_epoch` 후 loader 재생성**을 채택하고, `set_epoch`에서 persistent 감지 시 경고를 띄운다.

---

## 3. Collator 레이어

### 3.1 티처포싱 규약 (한 번만 선언, 안 그러면 이중 시프트)

collator는 **target 정렬**로 내보낸다. 1-스텝 자기회귀 시프트는 모델 책임. collator는
1칸 시프트를 절대 하지 않는다. 배치에 `target_aligned=True` 상수 키를 실어 트립와이어로 쓴다.

### 3.2 텐서 레이아웃

17개 스트림을 하나의 텐서로 합치지 않는다. Role Token이 **역할별 가산 벡터**이므로 모델은 role을
인덱스 규약이 아니라 실제 축으로 필요로 한다.
- 오디오: `(B, 2, 8, T)` — axis1: 0=self, 1=user
- 텍스트: `(B, T)` — **agent-only** (Moshi는 user 텍스트를 모델링하지 않는다)
- `2*8 + 1 = 17 = K_modeled` ✓

### 3.3 출력 배치 dict (핵심 키)

| key | dtype | shape | 의미 |
|---|---|---|---|
| `codes` | int64 | (B,2,8,T) | **delay 적용됨**. head-fill/batch-pad = `audio_init_id` |
| `role_ids` | int64 | (B,2) | 상수 `[[0,1]]`. role-shuffle ablation을 collator 플래그로 |
| `text_tokens` | int64 | (B,T) | delay 적용. 유효 구간 내 stream PAD/EPAD, 밖은 `batch_pad_text_id` |
| `stream_valid` | bool | (B,2,8,T) | **스트림별** 유효성 (§3.5 참조) |
| `attention_mask` | bool | (B,T) | 위치 수준 — 어떤 스트림이든 유효하면 True |
| `zone_ids` | uint8 | (B,T) | 0=A(system) 1=B(voice prompt) 2=C(dialogue) 3=batch pad |
| `audio_loss_weight` | fp32 | (B,2,8,T) | 곱하기만 하면 되는 가중치 |
| `text_loss_weight` | fp32 | (B,T) | 동상 |
| `teacher_topk_val` | fp16 | (B,2,T,k) | **raw pre-softmax**, codebook 0만 |
| `teacher_topk_idx` | int64 | (B,2,T,k) | 부재 시 `-1` (유효 코드가 아니므로 안전한 sentinel) |
| `kd_valid` | bool | (B,2,T) | `stream_valid[:,:,0]`에서 **파생** (재계산 금지) |
| `kd_frame_weight` | fp32 | (B,2,T) | 전이 프레임 가중 (§3.7) |
| `sample_type_id` / `lang_id` | int64 | (B,) | loss 라우팅. `en_solo`는 (type=ko_tts, lang=en) |
| `delay_offsets` | int64 | (9,) | 이번 배치에 실제 적용된 정규화 오프셋 — **반드시 로깅** |

메모리 주의: teacher가 지배적 (`B×2×T×32×10` bytes ≈ B=8,T=1500에서 7.7MB/배치).
병목이면 `teacher_topk_idx`만 int32로 — int64 일괄 적용을 깰 가치가 있는 유일한 지점.

### 3.4 필수 config assert

```python
for f in ("text_pad_id","text_epad_id","batch_pad_text_id",
          "audio_init_id","silence_codes","mimi_ckpt_id"):
    assert getattr(cfg, f) is not None, f"{f} unset (configs/tokens.yaml)"
assert cfg.text_pad_id != cfg.batch_pad_text_id   # RISKS §7.12
assert cfg.text_epad_id not in (cfg.text_pad_id, cfg.batch_pad_text_id)
```

`text_delay_sec → frames`는 **열린 결정**: 0.6s × 12.5Hz = 7.5프레임으로 정수가 아니다.
`round()` 후 `abs(sec*12.5 - round) < 0.25` assert → 0.6에서 큰 소리로 실패시켜 팀이 7이냐 8이냐를
명시적으로 고르게 한다.

`set_delay(DelayConfig)`를 노출해 Phase 1→2의 τ=2→1, text_delay→0 전환을 DataLoader 재생성 없이.

### 3.5 패딩 — 배치 최댓값 우右패딩 (`pad_to_multiple_of=8`), 고정 버킷 아님

`en_kd`는 1500프레임까지, `ko_tts`는 40~300프레임. Phase 2는 둘을 매 배치에 섞는다
(TRAINING_CURRICULUM). 고정 버킷은 짧은 쪽에 ~5× 낭비다. 패딩 낭비는 collator가 아니라
**sampler의 길이 버킷팅**이 억제한다 (§4.4).

**delay와 경계의 상호작용 — `stream_valid`가 (B,2,8,T)인 진짜 이유:**
delay 후에는 같은 행 안에서도 codebook마다 유효 구간이 다르다. 원 길이 `L`인 샘플은
`[0, L+max_off)`를 차지하고, codebook `k`는 `[off_k, off_k+L)`에서만 유효하다.
**단일 (B,T) 패딩 마스크로는 이걸 표현할 수 없고, 쓰면 양 끝에서 조용히 틀린다** —
head에서는 `audio_init_id`를 진짜 데이터인 양 학습하고, tail에서는 예측 가능한 `L-τ` 프레임을
날린다. `attention_mask`(B,T)는 합집합으로 어텐션용, **모든 loss 마스킹은 `stream_valid` 경유.**

`edge_mode="extend"` 기본 (감독 신호를 버리지 않음, τ≤2 추가 위치는 공짜).
`"truncate"`는 참조 구현 대조와 왕복 테스트용으로 유지.

### 3.6 delay 적용 알고리즘

```python
raw = [text_delay_frames] + [semantic_delay] + [acoustic_delay]*7   # len 9
off = [d - min(raw) for d in raw]     # 정규화: 모든 오프셋 ≥ 0
max_off = max(off)
```

**이 정규화 단계가 RISKS §7.8이 사는 곳이다.** 음수 text_delay(텍스트 선행, 프리트레인 ±0.6 영역)는
텍스트를 뒤로 미는 게 아니라 — 0번 위치 앞으로는 아무것도 못 민다 — **나머지 전부를 앞으로 민다.**
즉 `semantic_delay==0`이어도 codebook 0의 유효 오프셋이 0이 아니게 된다.
"cb0 오프셋은 0이니 teacher는 시프트 불필요"라고 가정한 코드는 파인튜닝에서는 맞고
프리트레인에서는 **KD 타깃을 학생 예측보다 정확히 한 프레임 앞서게** 만든다.

teacher는 **동일한 `off[1]`**을 쓴다. 두 호출 지점이 같은 헬퍼 `_place(dst, src, off, L)`를
공유해 발산이 구조적으로 불가능하게 하고, `kd_valid`는 `valid_audio[:,0]`에서 **파생**한다.
재계산이 §7.8 버그의 산실이다.

**작동 예 — K=3(text, cb0, cb1), τ=1, T=5, text_delay=0**
`raw=[0,0,1]` → `off=[0,0,1]`, `T_out=6`

| stream | off | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|---|
| text | 0 | w0 | w1 | w2 | w3 | w4 | ·|
| cb0 | 0 | a0 | a1 | a2 | a3 | a4 | ·|
| cb1 | 1 | init· | b0 | b1 | b2 | b3 | b4 |
| teacher(cb0) | 0 | P0 | P1 | P2 | P3 | P4 | ·|

**함정을 드러내는 변형 — 같은 τ=1, `text_delay = -1`**
`raw=[-1,0,1]` → `off=[0,1,2]`, `T_out=7`

| stream | off | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|---|
| text | 0 | w0 | w1 | w2 | w3 | w4 | ·| ·|
| cb0 | 1 | ·| a0 | a1 | a2 | a3 | a4 | ·|
| cb1 | 2 | ·| ·| b0 | b1 | b2 | b3 | b4 |
| teacher(cb0) | **1** | ·| P0 | P1 | P2 | P3 | P4 | ·|

`semantic_delay`가 여전히 0인데도 teacher 행이 1칸 움직였다. 여기서 `o0=0`을 하드코딩하면
**약 80ms의 턴테이킹 타이밍 왜곡이 학습되고 수렴하며 에러를 내지 않는다.**

### 3.7 loss weight — 텐서 3개 (하나도, 다섯도 아님)

출력 헤드별로 나눈다. 오디오 CE는 `(B,2,8,T,card)`, 텍스트 CE는 `(B,T,V_qwen)`, KD-KL은
`(B,2,T,card)` 위에서 돈다 — 어휘 크기가 달라 단일 텐서로 합치면 ragged concat이 된다.

```python
w_audio = stream_valid.float()                        # batch pad + delay 경계
w_audio *= (zone_ids != ZONE_A).float()[:,None,None,:]  # Zone A → 0
w_audio[:, :, 1:, :] *= 0.02                          # non-semantic codebook 1..7
w_audio *= synthetic_user_mask                        # 합성 무음 user 채널은 0

w_text = text_valid.float() * (zone_ids != ZONE_A).float()
w_text = where(text_tokens == PAD,  w_text * 0.30, w_text)
w_text = where(text_tokens == EPAD, w_text * 1.00, w_text)

kd_w = kd_valid.float() * kd_frame_weight
```

- **곱셈, 고정 순서, 유효성 마스크에서 시작.** batch pad는 별도 가중치가 아니라 `stream_valid`의
  0.0 바닥 — 하류 가중치가 실수로 패딩 위치를 부활시킬 수 없다.
- **EPAD는 다운웨이트하지 않는다.** ×0.3은 PAD의 ~65% 점유에 대한 클래스 불균형 보정이고,
  EPAD는 희귀하며 **발화 개시 트리거**다. 낮추면 이 프로젝트가 전이시키려는 바로 그 턴테이킹
  행동을 억누른다. `w_stream_epad_text`로 노출해 ablation 가능하게.
- **정규화는 loss의 일 (`sum(w*ce)/sum(w)`).** collator가 하면 신호별 그래디언트 노름 모니터링이
  해석 불가능해진다.

**스키마 갭 기록**: Zone A 길이를 행 메타로 실을 `zone_a_frames` 필드가 `arrow_features()`에 없다.
추가 필요.

### 3.8 KD 전이 프레임 가중 — **collator에서** (기본값)

텍스트 스트림이 가장 깨끗한 전이 신호이고, `en_kd`는 양쪽 모두 프레임 정렬 텍스트를 저장한다.
`find_solo_windows` ([en_solo_dataset.py:26](../../data_pipeline/datasets/en_solo_dataset.py#L26))가
이미 같은 신호로 활동 구간을 잡으므로 정의를 재사용한다 (두 번째 정의를 만들지 말 것).

```python
active = (text != PAD) & (text != EPAD) & text_valid
trans  = pad(onset(active) | offset(active))
trans  = max_pool1d(trans.float(), 2*hw+1, stride=1, padding=hw) > 0   # ±6프레임(~0.5s)
kd_frame_weight = 1.0 + (w - 1.0) * trans
```

collator에 두는 이유: ① 행이 정해지면 결정되는 *데이터* 속성이라 매 스텝 GPU 재계산은 낭비,
② **pre-delay 프레임 좌표**에서 전이를 정의한 뒤 `off[1]`로 밀어야 cb0와 정렬되는데 `off`를 아는 건
collator뿐 — loss에서 하면 delay 로직을 두 번 재구성하게 되고 그게 전형적인 §7.8 버그다,
③ 별도 `(B,2,T)` 텐서로 두면 "전이 프레임 KD 손실 vs 정상 프레임"을 분리 로깅할 수 있고
이게 Phase 2의 주요 진단이다.

### 3.9 정렬 자가검증 (디버그 모드) — §7.8 탐지기

**(a) 구조 왕복**: `codes[b,r,k,o:o+L] == raw[b][r,k,:L]`, head/tail은 `stream_valid` False,
`kd_valid == stream_valid[:,:,0]`.

**(b) 의미 시프트 스캔 — 진짜 탐지기.** 저장된 샘플 토큰은 그 프레임의 teacher top-k 안에
있어야 한다 (그 logits에서 뽑았으므로). `topk_dump=32` / `gen_top_k=250`이라 적중률이 1.0은
아니므로 **상대 시프트에 대한 argmax**로 assert한다.

```python
rates = {d: hit_rate(d) for d in (-2,-1,0,1,2)}
assert max(rates, key=rates.get) == 0, f"§7.8 misalignment: peak at {…}, off={delay_offsets}"
assert rates[0] > cfg.min_kd_hit_rate    # 기본 0.5, 전면 손상 탐지
```

ingest에서 캡처→프레임 매핑을 푸는 데 이미 쓰인 경험적 오프셋 논증
([en_kd_dataset.py:306](../../data_pipeline/datasets/en_kd_dataset.py#L306))을 **solver가 아니라
guard로 재사용**한다. `(B,2,T,32)` 위 마스크 비교 5회 — 1ms 미만. 잘못된 `o0`, 음수 text delay에서
빠뜨린 정규화, 크롭 슬라이스 off-by-one, 모델의 이중 시프트를 전부 잡는다.

---

## 4. Sampler / 믹싱 레이어

### 4.1 batch sampler, index sampler 아님

`Sampler[int]`로는 "이 배치 전체가 한 group" 도 "토큰 예산까지 채우기"도 표현 못 한다.
둘 다 필수 요구사항(§4.3, §4.4)이므로 `list[int]`를 yield하고 `batch_sampler=`로 넘긴다.

```
MixedArrowDataset (ConcatDataset)
  ▲ GroupIndex            # group별 전역 인덱스 범위 + cost 배열
  ▲ MixingBatchSampler    # step → group → 토큰 예산 배치
```

### 4.2 length-defined 에폭

`__len__ == steps_per_epoch`. 의미 있는 결합 에폭이 없다 — `ko_tts`는 `en_kd`보다 한 자릿수 크다.
"ConcatDataset 소진"은 믹싱 비율을 커리큘럼이 아니라 코퍼스 크기가 지배하게 만든다.
→ **커리큘럼은 global step 기준. epoch는 장부용.** LR/비율/체크포인트 전부 `global_step` 키.

### 4.3 뽑기 규칙

```
w = schedule.weights_at(s)
g = rng(seed, s // grad_accum).choice(w.keys(), p=normalize(w.values()))
batch = group_stream[g].next_batch(token_budget, max_batch, bucket_width)
```

- **멀티노미얼**, 결정적 인터리브 아님 (결정적이면 group이 step 패리티와 상관되고 램프가 불연속).
- **group은 grad_accum 윈도우 내 고정.** 안 그러면 한 옵티마이저 스텝의 손실 구성이 런마다 달라진다.
- **가중치는 *스텝* 비율**이지 샘플/프레임 비율이 아니다. 토큰 예산 배치이므로 스텝 비율 ≈ 프레임
  비율이지만 **샘플 비율과는 크게 다르다.** `realized_ratios()`가 스텝 점유와 프레임 점유를 둘 다
  보고해 이 괴리가 놀라움이 아니라 관측치가 되게 한다.

### 4.4 토큰 예산 배치 + 길이 버킷팅

T가 250~1500으로 6× 변동한다. 고정 B는 최악에 맞춰야 하므로 짧은 배치에서 메모리의 1/6만 쓰고,
토큰/스텝이 6× 요동쳐 **구성이 동시에 변하는 믹싱 스케줄과 나쁘게 상호작용**한다.

```
accept row i if (len(batch)+1) * max(T_max, T_i) <= token_budget  and  len+1 <= max_batch
```
`sum(T_i)`가 아니라 **패딩 비용 `B*T_max`** — 이게 실제로 활성화 메모리를 묶는 양이다.
버킷팅이 `B*T_max ≈ sum(T_i)`를 유지시킨다.

버킷팅은 **group 내부 전용**. `text_anchor`(오디오 없음)와 `ko_tts`(오디오 있음)를 한 배치에
섞으면 §4.5의 모듈 발산 문제가 rank *내부*로 재유입된다.

group-epoch당: 셔플 → `pool_multiplier(64) × world_size` 배치 분량 풀 → `num_frames//bucket_width`로
정렬 → 예산 그리디 커팅 → **배치 리스트 재셔플** → 배치 `k`를 rank `k % world_size`에.
재셔플이 중요하다: 없으면 grad_accum 윈도우 전체가 긴 꼬리에 몰릴 수 있고, 긴 샘플 스텝은
OOM 위험과 그래디언트 노름 이상치가 겹치는 바로 그 지점이다.

`text_anchor`는 `num_frames=0`이므로 `GroupIndex`는 `num_frames`가 아니라 **`cost`** 배열을 든다
(`cost = num_frames if > 0 else len(text_tokens_a)`), 매핑은 `loader.yaml`에 선언.

4×A100 시작값: `token_budget: 6000`, `max_batch: 16`, `bucket_width: 100`, `grad_accum: 4`.
튜닝 노브는 `token_budget` 하나.

### 4.5 분산 정확성 — **모든 rank가 같은 group을 뽑아야 한다** (핵심 제약)

두 축을 혼동하지 말 것:
- **어느 group**: `(seed, s // grad_accum)`만으로 시드. rank 무관 → 통신 0으로 전 rank 동일.
- **group 내 어느 행**: rank로 샤딩. 배치 커팅이 `perm_g`·`cost`·예산의 순수 함수라 rank 불변이므로,
  모든 rank가 같은 커팅을 계산하고 자기 슬라이스만 가져간다. collective 불필요.

**`group_sync: "global"` 강제. `world_size > 1`에서 `per_rank`면 팩토리가 raise.**

실패 모드가 구체적으로 나쁘다. FSDP2는 모듈 순회 순서로 키가 잡히는 NCCL collective를 삽입한다.
`text_anchor` 배치는 codes가 아예 없어 Depth Transformer / 오디오 임베딩 / `linears.0~7`을
호출하지 않는다. rank 0이 `text_anchor`를, rank 1이 `en_kd`를 뽑으면 rank 1은 rank 0이 절대 내지
않는 `all_gather`를 낸다. 결과: **GPU 사용률 100%로 에러 없이 행(hang)**, 워치독이 10~30분 뒤
엉뚱한 줄을 가리키며 죽는다. 그리고 **간헐적이다** — 멀티노미얼이 갈리는 스텝에서만 터지므로
100스텝 스모크 테스트는 통과한다.

부차적이지만 각각 독립적으로 충분한 이유 둘: ① FSDP2가 rank 간 그래디언트를 평균하므로 rank별
group이 다르면 신호별 유효 가중치가 *랜덤* rank 배정의 함수가 되어 설정한 loss 가중치가
얻는 가중치와 달라지고 가중치 ablation이 측정 불가능해진다. ② 토큰 가중치 정규화의 분모가
전역이어야 하는데, 동질 group이면 지역 분모가 전역적으로 옳다.

**단언으로 강제** (`debug.assert_group_sync`, 첫 200스텝 + 이후 1000스텝마다):
```python
t = torch.tensor([name_idx(group), step], device="cuda")
dist.all_gather(gathered, t)
assert all(torch.equal(g, t) for g in gathered), f"rank group desync at {step}"
```
2원소 all_gather 하나로 30분 hang을 즉시 읽히는 크래시로 바꾼다. 자체가 collective이므로
반드시 조건문 **바깥**에 둘 것.

### 4.6 num_workers / set_epoch

batch sampler는 **메인 프로세스**에서 돌므로 워커 수가 group 선택·결정성에 영향을 주지 않는다.
남는 위험 둘:
- Arrow mmap은 워커별 핸들 필요 (fork 시 공유 offset 누수). `get_worker_info()` 키 캐시로 지연 오픈.
- **prefetch가 스케줄 경계를 넘는다**: `num_workers=6, prefetch=2`면 최대 12배치가 in-flight이고
  그 group은 *sampler의* step 카운터로 정해졌다. 4000스텝 램프에서 <0.5% 오차 — 허용 가능하지만,
  체크포인트에 저장할 것은 **sampler의 step 카운터**이고 §6의 통계 테스트 허용오차가 이를 흡수해야 한다.

`DistributedSampler`는 쓰지 않는다 (`MixingBatchSampler`가 포섭). 재개 시
`set_step(global_step)` + `set_epoch(epoch)` + **`group_cursor` / `group_epoch` 복원**.
커서 없이 재개하면 모든 group 스트림이 조용히 0에서 재시작해 작은 group의 앞부분 N개를 다시 보여준다.
`state_dict()/load_state_dict()` 노출.

`epoch`는 group별 순열 시드에만 영향을 주고 **group 선택 RNG에는 영향을 주지 않는다** —
안 그러면 에폭 중간 재개가 원본과 다른 혼합을 재생한다.

### 4.7 group 소진 — `reshuffle` (복원추출 아님)

`en_kd`가 희소하면서(생성 1000대화 + 품질 필터) 동시에 Phase 1에서 ~93%, Phase 2에서 55~85%의
최고 가중치 group이다. 공격적으로 재활용된다.

복원추출은 노출 횟수를 푸아송 변수로 만든다 — 기대 8회에서 어떤 행은 14회, 어떤 행은 3회.
이미 과적합 위험이 있는 1000대화 합성 코퍼스에서는 순수 downside다. 비복원은 모든 행에
정확히 `⌊n⌋` 또는 `⌈n⌉` 노출을 준다.

**과적합 위험은 숨기지 말고 드러낸다:**
- `sampler/epochs/{group}`, `sampler/repeat_factor/{group}` 로깅.
- `max_repeat_factor` 초과 시 **group당 1회 경고, crash 아님** — 새벽 3시에 죽으면 안 되지만
  숫자는 체크포인트 B의 영어 held-out 곡선 옆에 찍혀야 "영어가 나빠졌다"와 "en_kd를 외웠다"를
  구분할 수 있다.
- 탐지기는 probe split: 믹싱 가중치 고정 상태에서 `en_kd`의 train/probe 격차 상승 = 암기.
- **caveat**: `en_solo`는 같은 아티팩트를 재크롭한 것이라 독립 데이터가 **아니다**.
  repeat factor를 `en_kd`와 함께 읽어야 한다. 지표 옆에 명시.

---

## 5. 스케줄 / Config

### 5.1 한국어 비율 램프 — step-keyed 구간선형, 정규화 안 된 가중치

단계형(10%까지 20k, 그다음 30%)은 커리큘럼이 **임계 기간**이라 부르는 바로 그 지점에 두 번의
불연속을 만든다. 10%→30% 점프는 손실을 튀게 할 만큼 큰 분포 이동이고, 체크포인트 B가 영어 붕괴를
보는 시점에서 **영어 저하가 한국어 *존재* 때문인지 *점프* 때문인지 구분 불가능**하게 만든다.
선형 램프는 간섭 곡선을 미분 가능하고 귀속 가능하게 한다. 단계형은 같은 step에 앵커 2개를 두면
표현되므로 스키마가 더 일반적이다.

```yaml
data:
  mix:
    interp: linear
    unit: global_step
    anchors:
      - {at: 0,     weights: {en_kd: 0.85, en_solo: 0.05, ko_tts: 0.05, text_anchor: 0.05}}
      - {at: 4000,  weights: {en_kd: 0.75, en_solo: 0.10, ko_tts: 0.10, text_anchor: 0.05}}
      - {at: 20000, weights: {en_kd: 0.55, en_solo: 0.10, ko_tts: 0.30, text_anchor: 0.05}}
      - {at: 45000, weights: {en_kd: 0.35, en_solo: 0.10, ko_tts: 0.50, text_anchor: 0.05}}
    constraints:
      text_anchor: {min: 0.03, max: 0.08}
      require_groups: [en_kd, ko_tts, text_anchor]
```

구성 시점 검증 (step 45000이 아니라 빌드에서 실패): 모든 앵커가 **동일한 group 집합**을 명명할 것
— 키 누락은 암묵적 0이 아니라 에러다. 암묵적 0이 바로 한 phase가 조용히 `text_anchor`를 잃는
경로이고, 그건 Phase 4를 무의미하게 만든다. 제약은 앵커에서만이 아니라 `range(0, max_steps, 100)`
조밀 스윕으로 검사. 첫 앵커 이전/마지막 이후는 클램프 (오버런 잡이 80% 한국어로 외삽하지 않도록).

### 5.2 config 트리

```
configs/
  tokens.yaml          ★ PAD/EPAD/id의 유일한 진실 원천
  paths.yaml
  data/
    text_tok.yaml      # TextTokCfg  (mixins.py:37)
    generation.yaml    # GenConfig
    filter.yaml        # FilterConfig
    prepare.yaml       # prepare_dataset.main()
    loader.yaml        ★ NEW — dataset + collator + sampler
    probe.yaml         # 평가 오버라이드
  model/haan_qwen3_8b.yaml
  train/{_base,phase1_warmup,phase2_joint,phase3_main,phase3_5_acoustic_graft,phase4_ja_lora}.yaml
  ablation/{curriculum_timing,kd_scope,codebook_init}.yaml
```

### 5.3 PAD/EPAD 주입 메커니즘

세 파일이 같은 id를 필요로 한다: `text_tok.yaml`(정렬), `filter.yaml`(PAD 비율 필터),
`loader.yaml`(collator 마스킹). **매직 정수를 3중 복제하는 것이 "데이터를 구울 때 쓴 id"와
"손실을 가중할 때 쓴 id"의 조용한 불일치가 출하되는 정확한 경로다.** 그리고 그 실패는 보이지 않는다
— 모델이 그냥 턴 경계를 학습하지 않을 뿐.

→ **`configs/tokens.yaml`이 id가 리터럴인 유일한 곳.**
- `loader.yaml`/`train/*.yaml`은 OmegaConf `${tokens.text_pad_id}` 보간.
- `text_tok.yaml`/`filter.yaml`은 현재 bare `yaml.safe_load` → dataclass 경로라 보간이 안 된다.
  호출부를 바꾸는 대신 **평문 유지 + 교차 파일 일관성 테스트**(§6.1)로 드리프트를 조용한 버그가
  아니라 빨간 테스트로 만든다. 오프라인 파이프라인의 무의존 로딩을 보존하는 선택.
- 현재 값은 전부 `null`이고 모든 소비자가 이미 큰 소리로 assert한다. collator가 세 번째 assert를 추가.

```yaml
# configs/tokens.yaml
# PAD/EPAD는 Qwen3 예약 슬롯에 새로 배정. <|im_end|>/<|im_start|> 재사용 금지.
# stream PAD(예측 대상, ×0.3) ≠ batch pad(완전 마스킹). 반드시 구분.
tokenizer_name: Qwen/Qwen3-8B
text_pad_id:  null    # TODO
text_epad_id: null    # TODO
batch_pad_id: null    # TODO — text_pad_id와 반드시 달라야 함
audio_pad_id: null
codebook_size: 2048   # schema.py:29와 로드 시 동등 assert
num_codebooks: 8      # schema.py:28
```

### 5.4 `text_anchor`는 별도 collator 경로

**`TextAnchorCollator`를 두고, text_anchor 샘플은 오디오 배치에 절대 넣지 않는다.**
혼합은 배치 수준이 아니라 **스텝 수준**(grad_accum 윈도우 내 마이크로배치 교대).

1. **`T`의 의미가 다르다.** 오디오는 12.5Hz 프레임, anchor는 토큰(max 2048). 축을 공유하면
   2048토큰 anchor 하나가 배치의 모든 오디오 행을 2048프레임(164초)까지 패딩시킨다 — 신호 0에
   연산 ~7× 팽창.
2. **delay 패턴이 정의되지 않는다.** 오프셋을 걸 codebook 축이 없다.
3. **`codes_a`가 `zeros((8,0))`.** 공유 shape를 맞추려 2048프레임×2역할×8코드북의 가짜 무음을
   실체화하면 샘플당 ~33k 날조 토큰이고, §2.5와 같은 실패 모드가 두 자릿수 크게 온다.
4. **loss 라우팅이 완전 분리.** PAD/EPAD 의미도, Zone A도, 오디오 브랜치도, KD도 없다.
   §3.7의 모든 규칙에 `if sample_type == text_anchor` 가드가 붙어야 한다.
5. anchor의 임무는 파국적 망각 방지이고 가중치는 독립 튜닝 대상이다. 별도 배치라야
   "anchor loss"가 혼합 손실의 마스킹된 부분집합이 아니라 직접 로깅 가능한 스칼라가 된다.

`sample_type_id`는 오디오 배치에도 계속 실어(값 2가 거기 없을 뿐) loss가 라우팅을 가정이 아니라
단언할 수 있게 한다.

### 5.5 `build_dataloader`

```python
def build_dataloader(cfg, split, rank, world_size) -> tuple[DataLoader, MixingBatchSampler | None]:
    """sampler를 함께 반환한다 — persistent_workers 하에서 DataLoader.batch_sampler로
    도달할 수 없는데 트레이너가 set_epoch/set_step/state_dict를 호출해야 하기 때문."""
```
순서 (각 단계는 우아한 저하 대신 큰 소리로 실패):
① config resolve → `_assert_token_ids` (**I/O 이전**) ② group `load_from_disk` + 비어있지 않음 assert
+ mix 앵커 group ⊆ 로드된 group ③ `GroupIndex` (`cost` 배열만 읽음, `codes_*`/`teacher_*` 미접촉)
④ `MixedArrowDataset` ⑤ `MixSchedule` 검증 ⑥ `MixingBatchSampler` + `group_sync` assert
⑦ group-aware collator ⑧ `DataLoader(batch_sampler=, collate_fn=)`
(`batch_size`/`shuffle`/`sampler`/`drop_last` 전부 미설정이어야 함)

### 5.6 probe 로더 — 플래그가 아니라 다른 경로

| 항목 | train | probe | 이유 |
|---|---|---|---|
| sampler | `MixingBatchSampler` | `GroupSequentialBatchSampler` | 지표는 **group별**이어야 한다. 혼합 평균은 측정하려는 트레이드오프를 정확히 가린다 |
| 순서 | 셔플 | 인덱스 오름차순 | 체크포인트 간 배치 구성이 바이트 동일 ⇒ 지표 델타 = 모델 델타 |
| crop | random | **none** | 랜덤 크롭 분산이 체크포인트 B/C가 보는 작은 델타를 삼킨다 |
| A/B swap | 0.5 | **0.0** | 동일 |
| 샤딩 | `k % ws` | + `drop_last=False` + **중복 패딩 배치**(`is_pad_batch=True`, 리덕션 전 마스킹) | 모든 probe 행을 정확히 1회 평가하면서 rank별 collective 수를 맞춘다 (§4.5와 같은 제약) |

구조적 가드 2개: 팩토리가 `split=="probe"` ⇒ 옵티마이저 미부착을 호출부에서 assert하고,
probe 데이터셋은 `read_only=True` 마커로 생성한다. probe/train uid 서로소는 `is_holdout` 덕에 공짜.

---

## 6. 테스트 계획

**Config 무결성** — ① `tokens.yaml`/`text_tok.yaml`/`filter.yaml` id 일치 (§5.3 미러링을 안전하게
하는 가드) ② id `null`일 때 세 소비자 전부 config 파일명을 담은 `AssertionError`
③ stream PAD == batch pad면 raise ④ `REGISTRY`의 모든 `source`가 `loader.yaml:groups`에 존재
⑤ 스케줄 검증(비단조 앵커, at:0 누락, 앵커 간 group 집합 불일치, 앵커 *사이*에서 제약 위반)

**Dataset** — ⑥ 아이템 키 정확히 일치, dtype/shape ⑦ `codes_self.shape[1] == text_self.shape[0] == num_frames`
⑧ **fast path가 `row_to_arrays`와 elementwise 일치** (zero-copy 읽기의 유일한 오라클)
⑨ `datasets.Dataset.__getitem__`을 raise로 몽키패치해도 동작 (전체 실체화 회귀 방지)
⑩ solo가 `silence_codes`로 채워짐 (0이 아님) ⑪ `en_solo`가 (source=en_solo, type=ko_tts, lang=en)
⑫ text_anchor T=0 계약 ⑬ `len` 2배 & solo는 안 됨 ⑭ swap involution
⑮ **swap 후 teacher가 `_b`를 따라감** (파일에서 가장 가치 높은 테스트)
⑯ 에폭당 모든 uid가 서로 다른 `swapped`로 정확히 2회 ⑰ 크롭 정렬 (ramp 심어 (K,T)와 (K,T,topk)
슬라이스 경로 off-by-one 검출) ⑱ probe 크롭 결정적 ⑲ 크롭 출력 contiguous (부모 버퍼 유지 검출)
⑳ 워커 수 {0,2,4}에서 크롭 동일 ㉑ `__getitem__`이 전역 RNG 미접촉 ㉒ picklable & 페이로드 작음
㉓ **teacher 값이 raw** (누가 친절하게 softmax 넣는 것 방지) ㉔ **delay 미적용** (ramp로 검증)

**Collator** — ㉕ **delay 왕복 역변환** (τ∈{0,1,2} × text_delay∈{−8,0,+8}) — 핵심 테스트
㉖ 오프셋 정규화 비음수 (§3.6 예시 하드코딩 테이블 대조)
㉗ **teacher가 cb0와 동일하게 시프트** (음수 text delay 포함 — *하드코딩 `o0=0` 버그에서 실패*)
㉘ 시프트 스캔이 0에서 피크 + **뮤테이션 테스트**(teacher를 1 굴리면 검사가 raise해야 함)
㉙ batch pad 가중치 정확히 0 / stream PAD 0.3 / EPAD 1.0 / Zone A 0.0
㉚ Zone A 음향 프레임은 0.0이지 0.02가 아님 ㉛ `text_delay_sec=0.6`이 raise
㉜ text_anchor 행을 `KDCollator`에 주면 `TextAnchorCollator`를 가리키며 raise

**Sampler / 분산** — ㉝ `len == steps_per_epoch` ㉞ 토큰 예산 준수 + 패딩 낭비 < 0.15
㉟ 배치가 group 혼합 안 함 ㊱ grad_accum 윈도우 내 group 고정
㊲ **rank 0~3이 10k스텝 동안 동일 group 시퀀스** (§4.5 hang의 회귀 테스트)
㊳ rank 파티션 서로소 & 완전 ㊴ `set_epoch`가 행은 바꾸되 group 시퀀스는 유지
㊵ `state_dict` 재개가 배치 500~1000을 인덱스 단위로 재현 (group 커서 포함)
㊶ 재활용 노출 균일도 ±1, 복원추출 모드에서는 같은 단언이 *실패*해야 함 (두 모드가 조용히
동일해지는 것 방지)

**통계적 비율 테스트 (별도 호출)**
```python
counts = Counter(s.group_at(t) for t in range(step, step + W))   # W = 4000
n_updates = W // s.grad_accum
for g, p in schedule.mean_weights_over(step, step + W).items():
    se = sqrt(max(p*(1-p), 1e-6) / n_updates)
    assert abs(counts[g]/W - p) < 4*se + 0.01
```
① 목표는 `weights_at(step)`이 아니라 **`mean_weights_over`** — 램프 내부에서는 가중치가 창을 가로질러
움직이므로 좌단 비교는 버그처럼 보이는 계통 편향을 만든다. ② `n_updates = W // grad_accum`이 실제
독립 추출 수 (§4.3); `W`를 쓰면 허용오차가 너무 빡빡해 flaky. ③ `+0.01`이 §4.6의 prefetch 지연 흡수.
④ 동반 `test_realized_frame_ratios`를 **프레임 공간**에서 3× 느슨하게 — "스텝의 50%가 한국어"인데
한국어 배치가 8× 짧아 그래디언트 질량의 12%인 §4.3의 함정을 잡는 건 이쪽이다.
커리큘럼이 "한국어 비율"로 실제 의미하는 것은 프레임 점유다.

---

## 7. 구현 순서

0. **§9 준비 계층** — `ensure_prepared` + sentinel. `data/prepared/`가 없으면 나머지가 무의미
1. `item.py` + `config.py` + `configs/tokens.yaml` — 계약 먼저 못박기
2. `crop.py` + `dataset.py` + 테스트 ⑥~㉔ (`row_to_arrays` 오라클 대조가 관문)
3. **silence code 도출** (§2.5) + `mimi_silence.json` 커밋 — collator의 선행 조건
4. `collator.py` + 테스트 ㉕~㉜ (delay 왕복 + teacher 시프트 뮤테이션 테스트가 관문)
5. `schedule.py` + `sampler.py` + 테스트 ㉝~㊶ + 통계 테스트
6. `loader.py` + probe 경로 + `configs/data/loader.yaml`
7. ~~`zone_a_frames`를 `arrow_features()`에 추가~~ → **철회**: Zone A/B는 재계산 가능하므로
   collator가 구성한다 (§10.9). 스키마 범프 불필요.

**블로커 2개**:
- PAD/EPAD id 미확정. 1~2단계는 `null` 상태로 진행 가능(assert가 정확히 이 목적),
  3단계부터는 실제 값이 필요하다.
- `MoshiSelfTalkEngine`이 `NotImplementedError` 스텁
  ([en_kd_dataset.py:112](../../data_pipeline/datasets/en_kd_dataset.py#L112)) →
  `data/prepared/en_kd/`가 현재 생성 불가. §9.5의 A/B 경로 중 택일 필요.

---

## 8. 열린 결정 사항

| 항목 | 선택지 | 비고 |
|---|---|---|
| PAD/EPAD id | Qwen3 예약 슬롯 중 지정 | 전체 블로커 |
| `text_delay` 0.6s | 7 프레임 vs 8 프레임 | 12.5Hz에서 정수가 아님. assert로 강제 선택 |
| `token_budget` | 6000 시작 | OOM 스윕으로 확정 |
| `max_frames` (크롭) | 750 시작 | 생성 상한 1500과 메모리 사이 |
| silence: 상수 프레임 vs 루프 | 상수 시작 | 상수는 실제 무음과 자명하게 구별되어 "user 없음" 지름길을 준다. 루프(랜덤 위상)가 더 안전 |
| `teacher_topk_idx` dtype | int64 vs int32 | 메모리 병목이면 int32 |

---

## 9. 데이터 준비 계층 — `data_pipeline` 변경안

**목표**: Dataset 객체를 한 번 만들면 그 안에서 Moshi 대화 생성 → 저장 → Arrow 준비까지
보장된다. torchvision `MNIST(root, download=True)`와 같은 **ensure-materialized** 관용구.

### 9.1 왜 완전 병합은 안 되는가 (설계를 결정하는 제약)

"빌더 클래스 = 학습 Dataset 클래스"로 합치고 싶지만 두 가지가 막는다.

1. **N:1 관계.** [prepare_dataset.py:60-68](../../data_pipeline/prepare_dataset.py#L60-L68)에서
   `kss + css10_ko + zeroth_ko + common_voice_ko` → `ko_tts` 그룹 **하나**로 병합된다.
   학습 Dataset의 인덱스 공간은 4개 빌더에 걸쳐 있으므로 1:1 대응이 없다.
2. **빌더 간 DAG.** `en_solo`는 `en_kd` 아티팩트를 입력으로 받는다
   (`--root data/generated/en_kd`). 독립 단위가 아니라 순서 의존 그래프다.

→ **빌더는 빌더로 남기고, 학습 Dataset이 "준비 보장" 단계를 소유한다.**

### 9.2 진짜 문제는 이름이다

혼란의 원인은 `BaseDataset`이 `torch.utils.data.Dataset`이 아닌데 "Dataset"이라 불리는 것.
[base.py:24-26](../../data_pipeline/datasets/base.py#L24-L26)의 docstring이 이미 둘이 다르다고
말하고 있지만 이름이 그걸 가린다.

| 현재 | 변경 후 | 이유 |
|---|---|---|
| `BaseDataset` | **`SourceBuilder`** | torch Dataset이 아님. `build()`/`iter_samples()`만 가짐 |
| `AudioSourceDataset` | `AudioSourceBuilder` | 동상 |
| `*Dataset` (kss 등) | `*Builder` | 동상 |
| `build_dataset()` | `build_source()` | |
| `REGISTRY` | 유지 | CLI 키 그대로 |

`Dataset`이라는 이름은 `training/data/`의 torch Dataset만 쓴다. 순수 rename이라 위험도 낮고,
이 이름 충돌이 사라지면 "왜 두 개냐"는 질문 자체가 없어진다.

### 9.3 `ensure_prepared` — 멱등 준비 함수

`prepare_dataset.main()`을 재사용 가능한 함수로 리팩터한다.

```python
# data_pipeline/prepare_dataset.py
def ensure_prepared(group: str, cfg: dict, *, force: bool = False) -> Path:
    """data/prepared/{group}/ 을 보장한다. 이미 있으면 즉시 반환 (멱등).

    1. sentinel 확인 → 있으면 return
    2. 파일 락 획득 (동시 프로세스 이중 생성 방지)
    3. 락 획득 후 sentinel 재확인 (경합 시 다른 프로세스가 이미 끝냈을 수 있음)
    4. group에 속한 SourceBuilder 전부 build() → iter_samples() → write()
    5. sentinel 기록 (schema_version, git sha, builder 목록, 행 수)
    """
```

**sentinel**: `data/prepared/{group}/_SUCCESS.json`. HF의 `dataset_info.json` 존재만으로
판정하면 **중단된 `save_to_disk`의 부분 산출물을 완성본으로 오인**한다. 별도 sentinel을
마지막에 원자적으로 쓴다. 내용에 `schema_version`을 넣어 §3.7의 `zone_a_frames` 추가 같은
스키마 변경 시 자동 재빌드를 트리거한다.

**멱등성은 이미 하부에 있다**: `is_cached(uid)`
([mixins.py:167](../../data_pipeline/datasets/mixins.py#L167))가 인코딩을 건너뛰고,
`is_holdout`이 해시 기반이라 재실행해도 split이 동일하다. `ensure_prepared`는 그 위에
그룹 단위 게이트를 얹는 것뿐이다.

### 9.4 `MoshiKDDataset.__init__`에서의 호출 — 그리고 rank 가드

```python
def __init__(self, root, source, split="train", *,
             cfg, build: str = "never", ...):
    """build: "never" | "if_missing" | "force" """
    if build != "never":
        ensure_prepared(source, cfg.prepare, force=(build == "force"))
    ...  # Arrow 오픈
```

- **학습 기본값은 `"never"`** — 없으면 실행할 정확한 명령어를 담아 큰 소리로 실패.
- 준비 전용 진입점이 `"if_missing"`을 쓴다:
  ```bash
  python -m training.data.prepare --group en_kd     # ← "한 번 동작"이 이것
  ```
  같은 클래스, 같은 `__init__`, 플래그만 다르다.

**`world_size > 1`에서 무거운 그룹은 빌드를 거부한다.** rank0-only 빌드 + `barrier()`는
표준 패턴이지만 여기서는 위험하다: en_kd 자체 대화 생성은 **시간 단위**인데 NCCL 워치독
기본 타임아웃은 10~30분이다. rank 1~3이 barrier에서 대기하다 **워치독이 잡을 죽인다** —
그것도 "빌드가 오래 걸림"이 아니라 "collective timeout"으로 보고되어 원인 추적이 어렵다.

```python
HEAVY = {"en_kd", "en_solo", "ko_tts"}   # Mimi/Moshi GPU 인코딩 수반
assert not (build != "never" and world_size > 1 and source in HEAVY), (
    f"'{source}'는 학습 잡 안에서 빌드할 수 없다 (rank {world_size}개). "
    f"먼저 실행: python -m training.data.prepare --group {source}")
```
가벼운 `text_anchor`만 rank0 + barrier를 허용한다.

### 9.5 en_kd 생성 경로 — 택일 필요

`MoshiSelfTalkEngine.__init__`이 스텁이라 `_generate()`가 동작하지 않는다.

| | A. 엔진 구현 | B. Colab + ingest |
|---|---|---|
| 방식 | 로컬 A100에서 batch 2N 크로스피딩 자체 대화 | Colab 생성 → `ingest_ab_selfplay()` |
| 상태 | 팀 fork의 streaming 인터페이스 확정 대기 | **이미 동작** ([en_kd_dataset.py:277](../../data_pipeline/datasets/en_kd_dataset.py#L277)) |
| 규모 확장 | 자동 | 수동 |
| `ensure_prepared` 통합 | 완전 자동 | 수동 산출물 존재 확인 후 ingest |

B를 §9.3의 1단계로 넣고 A를 후속으로 두면 준비 계층을 지금 완성할 수 있다.
B 경로에서 `ensure_prepared`는 Colab 산출물 디렉토리가 없으면 "이 경로에 `dialogue_*.npz`를
두고 다시 실행하라"는 메시지로 실패한다 — 조용한 빈 데이터셋보다 낫다.

### 9.6 변경 요약

| 파일 | 변경 |
|---|---|
| `datasets/base.py` | `BaseDataset` → `SourceBuilder` rename (+ 하위 전부) |
| `datasets/*_dataset.py` | rename 반영. 파일명은 `*_builder.py`로 |
| `prepare_dataset.py` | `main()` → `ensure_prepared(group, cfg, force)` 추출, sentinel + 파일 락 |
| `datasets/en_kd_dataset.py` | `MoshiSelfTalkEngine` 구현(A) 또는 ingest 경로 문서화(B) |
| `schema.py` | `zone_a_frames` 필드 추가 (§3.7) + `SCHEMA_VERSION` 상수 |
| `training/data/prepare.py` | **신규** — 준비 전용 CLI 진입점 |
| `training/data/dataset.py` | `__init__(build=...)` 파라미터 + rank 가드 |

**테스트 추가**: ㊷ `ensure_prepared` 2회 호출 시 두 번째가 재빌드 없이 즉시 반환
㊸ sentinel 손상/부분 산출물에서 재빌드 트리거 ㊹ `schema_version` 불일치 시 재빌드
㊺ `world_size>1` + HEAVY 그룹 + `build!="never"` → 실행 명령어를 담고 raise
㊻ 동시 프로세스 2개가 같은 group을 `ensure_prepared` 해도 아티팩트가 1벌

---

## 10. 정정 — 구현이 계획을 반박한 지점

구현하면서 드러난 오류들. 원래 서술은 위에 그대로 두고 여기에 정정을 모은다.

### 10.1 §2.6 `ascontiguousarray`로는 mmap에서 분리되지 않는다

크롭 창이 행 전체를 덮으면(`T <= max_frames`, 즉 **모든 `ko_tts` 샘플과 짧은 `en_kd` 행 전부**)
슬라이스가 이미 contiguous라서 `np.ascontiguousarray`가 입력을 그대로 돌려준다. 텐서는 읽기 전용
mmap 컬럼의 view로 남고 부모 버퍼가 고정되며, `torch.from_numpy`가 "not writable ...
undefined behavior" 경고를 낸다.

역설적으로 **크롭된 경로만 우연히 안전하다** — strided 슬라이스가 복사를 강제하기 때문. 즉 버그가
§2.6이 보호하려던 바로 그 경로에 숨는다. `_materialize()`의 무조건 `np.array(copy=True, order="C")`로
교체했고 teacher fp16 캐스트를 같은 패스에 접었다(어차피 복사 1회라 추가 비용 없음).

### 10.2 §5.1 "앵커 사이 제약 위반"은 선형 보간에서 발생 불가

정규화되지 않은 가중치를 선형 보간하면 어떤 group의 정규화 점유율은 두 affine 함수의 비이고,
이는 각 구간 내부에서 단조다. 따라서 극값은 항상 앵커에 놓인다 — 양 끝 앵커가 범위 안이면 내부는
전부 안이다. 조밀 스윕은 유지했지만(`interp`에 비선형 모드가 생기는 순간 다시 필요해짐) 요구사항을
정직한 두 테스트로 분리했다: 실패가 앵커가 아니라 step 900에서 보고되는지, 그리고 `sweep_stride`를
키우면 같은 config가 step 1000에서야 걸리는지.

관련: §5.1은 단계형 스케줄을 "같은 step에 앵커 2개"로 표현한다고 했지만 이는 strictly-increasing
규칙과 모순된다. strictly-increasing을 채택했으므로 단계형은 `n`과 `n+1`로 인코딩해야 한다.

### 10.3 §5.5 반환 타입과 §5.6 crop 모드

- 반환은 `tuple[DataLoader, sampler]`가 아니라 **`LoaderBundle`**. `set_epoch`이 sampler뿐 아니라
  모든 dataset에도 전파돼야 하고 index/schedule도 트레이너가 붙잡고 있어야 한다.
- §5.6의 probe `crop: none`은 존재하지 않는 모드다. `crop_mode`는 `random|center|head`이고
  probe는 **`center`** 강제.

### 10.4 §5.5 ①의 "I/O 이전 토큰 검증"이 실제로는 안 됐다

`DataConfig.__post_init__`은 `silence_codes`와 `text_pad_id`만 요구한다. 나머지
(`batch_pad_id` / `text_epad_id` / `audio_init_id` / `mimi_ckpt_id`)는 5단계의 `KDCollator`
생성 시점에 검사되는데, 그때는 이미 각 group을 두 번씩(`MoshiKDDataset` + `GroupIndex.from_prepared`)
연 뒤다. tokens.yaml이 전부 null인 현재는 매 호출이 전체 인덱스 I/O를 치르고 나서야 "id가 없다"는
말을 듣는다. `build_dataloader` 진입부에서 6개를 한꺼번에 `require()`하도록 수정했다.

### 10.5 §3.4 필드명 / §9.3 시그니처

- §3.4·§3.3이 `batch_pad_text_id`라 쓰지만 실제 필드는 **`batch_pad_id`**.
- §9.3의 `ensure_prepared(group, cfg, *, force)`는 loader.py의 호출부와 맞지 않는다.
  실제 시그니처는 **`ensure_prepared(group, *, root, cfg=None, force=False)`**.

### 10.6 §3.3 teacher 텐서 메모리 추정이 2배 과다 — 그리고 그게 설계 질문을 가리킨다

§3.3은 teacher를 `B×2×T×32×10` 바이트로 잡지만, §2.3이 **self 쪽 teacher만 유지**한다
(1.5MB → 0.75MB 주장의 근거). 따라서 `(B,2,T,k)` 레이아웃은 후반부가 전부 `-1`로 보장된 role 축을
할당한다. 계약대로 구현했지만(loss가 균일한 role 축을 원할 수 있음) 정직한 선택지는 둘이다:
`(B,T,k)`로 줄이고 loss가 unsqueeze하거나, 아니면 §3.3의 추정치가 실내용의 2배임을 인정하거나.
B=8·T=1500·k=32에서 배치당 ~3.8MB의 상수 텐서가 워커 IPC를 넘는다 — §2.8이 int16→int64 캐스트를
거부한 바로 그 경계다. `teacher_topk_idx`가 §3.3이 예상한 병목이 되면 **이 축이 먼저 잘라야 할
더 싼 대상**이다.

### 10.7 `RoutingCollator`의 키 충돌 (구현에서 수정됨)

오디오 배치와 anchor 배치가 `text_tokens` / `text_loss_weight` / `sample_type_id` / `lang_id` /
`sample_uid` / `target_aligned`를 공유하는데 **의미가 다르다** — `text_tokens`가 오디오에서는
프레임 정렬 `(B, T_frames)`, anchor에서는 비정렬 `(B, L_tokens)`. `"text_tokens" in batch`로
분기하는 트레이너는 한쪽에서 조용히 틀린다. 명시적 **`batch_kind`** 키(`"audio"` | `"text_anchor"`)를
추가했다.

### 10.8 테스트가 실제로 잡은 것 (뮤테이션 검증)

통합 테스트는 첫 실행에 전부 통과했다 — 즉 통과 자체는 증거가 아니다. 뮤테이션을 넣어 검증했고,
초기 테스트를 **살아남은** 것이 셋 있었다:

- `GroupIndex`에서 `np.repeat(cost,2)` → `np.tile(cost,2)`: en_kd cost가 `[a,b,a,b]`가 되는데
  dataset의 `_resolve`는 `index//2` → `[a,a,b,b]`. **group 소속도 길이도 동일**해서 기존 assert가
  전부 통과한다. 토큰 예산이 틀린 길이로 배치를 잡아 패딩 낭비와 OOM이 나는데 가리킬 대상이 없다.
  → 전역 id 테스트가 `index.cost(g)[local] == 실제 행 길이`까지 확인하도록 강화.
- 커서를 `world_size`가 아니라 `1`씩 전진: 매 스텝 *내부*에서는 rank가 서로소라 안 보이고
  스텝 *사이*에서만 행이 재공급된다. 픽스처 코퍼스를 24/48/48행으로 키워야 관측 가능해졌다
  (group-epoch당 배치가 ~4개면 매 스텝 새 에폭으로 굴러가 버그가 드러나지 않는다).
- pad-batch 오분류는 `world_size=3`에서만 나타난다 → 1~4 전체를 parametrize.

**교훈**: 이 계층의 버그는 크래시가 아니라 "그럴듯한 통과"로 나타난다. 통합 테스트는 뮤테이션으로
검증하지 않으면 검증된 게 아니다.

### 10.9 §3.7의 `zone_a_frames` 스키마 갭 — 철회

Zone A(시스템 프롬프트)와 Zone B(voice prompt)를 행에 저장하려던 계획을 철회한다.
§2.5의 기준("불러올 때 다시 계산할 수 있으면 저장하지 않는다")을 적용하면 둘 다 저장 대상이
아니다 — 시스템 프롬프트는 고정 config이고, 참조 음성은 **같은 화자의 다른 행**이다.

저장했을 때의 손실이 크다: 참조 음성이 행마다 복제되어 코퍼스가 부풀고, 무엇보다 참조가
샘플에 고정되어 TRAINING_CURRICULUM Phase 2가 요구하는 **"샘플마다 다양한 참조 음성"**의
에폭별 변주가 사라진다.

채택: **collator가 `[Zone A | Zone B | Zone C]`를 조립**하고, Dataset이 `speaker`로 참조 행을
찾아 `ref_codes`/`ref_text`로 넘긴다. 스키마 범프 불필요.

주의점 하나 — teacher top-k는 Zone C 프레임에 정렬돼 있으므로 **prefix 길이만큼 추가로
시프트**해야 한다. 빠뜨리면 RISKS §7.8의 조용한 오정렬이고, 실측으로 12프레임 KD 스큐가 난다.

### 10.10 D1 결정 — Zone B 참조는 §7.4 논증 1(구조적)만 따른다

§7.4는 두 개의 별개 논증을 담고 있다.

1. **구조적 (본문)**: PersonaPlex는 voice prompt 구간 텍스트를 패딩해 "텍스트 없는 오디오"(OOD)를
   만든다. Haan은 전사를 넣어 그 구간을 정상 생성 구간과 **구조적으로 동일**하게 만든다.
2. **의미적 (별도 문단)**: 참조를 도메인 오프닝 인사로 구성해 의미 오염·프롬프트 인젝션을 막는다.

**채택: 1만 따른다.** 전사가 존재하기만 하면 구조 요건은 충족되고, 내용이 Zone C와 무관해도
무방하다. 2가 든 유일한 근거는 **인젝션 차단**인데 그건 추론 시점 문제이고, 학습 시 참조 풀은
이미 통제되어 있다.

참조 선택 기준은 **같은 화자의 다른 발화**로 유지한다 — 단 이는 구조 요건이 아니라 음색 클로닝
요건(§5.2)에서 온다. 내용 관련성은 통제하지 않는다.

**기록해 둘 잔여 리스크 (문서에 없음, 추론)**: Zone B와 Zone C가 매번 의미적으로 무관하면
모델이 "직전 맥락은 다음 발화와 무관하다"를 학습할 수 있다. 대화 모델에는 반대 교훈이다.
반대로 "voice prompt는 음색만, 내용은 무시"가 옳은 학습이라는 해석도 성립한다. 이 축은 문서가
다루지 않으며, Phase 2 평가 시 확인 대상이다.

**배포 시 요건 (이월)**: 참조 풀을 도메인 오프닝 인사로 통제해야 §7.4의 인젝션 경로가 닫힌다.
학습이 아니라 서빙 단계의 책임으로 이월한다.

### 10.11 D3·D4 결정 — Zone A는 전 소스, voice prompt는 48프레임

**D3: Zone A를 `en_kd`에도 붙인다** (`zone_a_sources = ("ko_tts", "en_kd")`).

teacher는 prefix 없이 대화를 생성했지만 그게 손실에 닿지 않는다 — Zone A는 완전 마스킹이고
`kd_valid=False`이며, teacher top-k는 prefix 길이만큼 시프트되어 자기 Zone C 프레임에 그대로
안착한다(뮤테이션 검증 완료).

제외하는 쪽이 오히려 위험하다. `ko_tts`·`en_solo`는 prefix를 갖고 `en_kd`만 없으면
**"Zone A 없음" = "턴테이킹 하라"** 가 완벽히 상관된다. 언어보다 깨끗한 지름길 신호이고,
이는 RISKS §1이 프로젝트 최대 위험으로 지목하고 `en_solo`가 깨려는 바로 그 상관이다.
배포 시 모든 시퀀스가 Zone A로 시작하는데 턴테이킹 회로만 Zone A 없는 맥락에서 학습되는
문제도 있다.

Zone B는 `ko_tts` 전용 유지 — en_kd 행은 두 목소리라 단일 화자 참조가 정의되지 않는다
(Dataset이 `speaker == ""`를 assert한다).

**D4: `voice_prompt_frames = 48`** (3.84초). `project_amnesty/datasets/scenarios/prep_data.py`의
`REF_FRAMES`와 동일 — 이 레포에서 실제로 클로닝을 끝까지 돌려본 값이다. 기존 100(8초)은 실제
Zeroth 코퍼스에서 Zone B가 Zone C와 맞먹어 컨텍스트 예산의 절반을 프롬프트가 먹었다.

**D2(시스템 프롬프트 문구)는 보류.** `system_prompt_ids`가 config 주입이라 배포 포맷 확정 후
바꿔도 코드 변경이 없다. 지금 중요한 건 길이(=마스킹 구간)뿐이다.

### 10.12 ② ASR 방향 — `ko_asr`를 같은 데이터 위의 별도 그룹으로

DATA_STRATEGY §4.2의 양방향 재활용. ARCHITECTURE §5.0.2가 범위를 결정한다:

> *"a single delay hyper-parameter allows for switching from an ASR to a TTS model
> **with no changes in the loss, architecture, or training data**"*

**delay만 바뀐다.** 같은 행, 같은 손실, 같은 채널 — §5.0.2는 ASR도 **self(에이전트) 오디오
채널을 재활용**한다고 명시하므로 user 스트림을 전사하는 게 아니다.

부호 규약(`DelayConfig.offsets()`에서 확인): **양수** `text_delay_frames` = 텍스트가 뒤로 =
오디오 선행 = ASR. 음수 = 텍스트 선행 = TTS.

**채택: `ko_asr`를 별도 믹싱 그룹으로.** 배치 태깅 대신 이걸 고른 이유는 ASR 비율을 기존
`MixSchedule` YAML로 표현할 수 있기 때문이다. 태깅 방식은 믹싱 축을 하나 더 만들어야 한다.

디스크 중복 없음 — `GROUP_ALIASES = {"ko_asr": "ko_tts"}`로 같은 디렉토리를 읽는다. 문자 그대로
같은 행이므로 `is_holdout` split이 이미 적용되어 있어 홀드아웃 문제가 발생하지 않는다.

**미정 값**: `asr_text_delay_frames = 8`. 문서에 ASR 전용 수치가 없다. Table 1의 ±0.6초를
쓰면 0.6×12.5 = 7.5프레임으로 정수가 아니라(`_frames_from_sec`가 거부) 8은 반올림 선택이며
실측 확인이 필요하다.

**남은 두 가지 (의도적 미처리)**:
- `ko_asr`는 probe split에 나타나지 않는다 — `build_dataloader`가 probe 그룹을 디렉토리
  존재로 발견하기 때문. ASR 방향 품질을 프로빙하려면 별도 결정이 필요하다.
- `build_source_datasets`의 하드코딩 `_SOURCES`에 `ko_asr`가 없다. `build_dataloader` 경로가
  아니라 무해하지만, 별칭 그룹을 절대 반환하지 않는다.
