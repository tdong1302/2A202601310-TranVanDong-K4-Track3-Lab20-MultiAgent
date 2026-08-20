# Lab Guide: Multi-Agent Research System

## Scenario

Bạn cần xây dựng một research assistant có thể nhận câu hỏi dài, tìm thông tin, phân tích và viết câu trả lời cuối cùng. Lab yêu cầu so sánh hai cách làm:

1. **Single-agent baseline**: một agent làm toàn bộ.
2. **Multi-agent workflow**: Supervisor điều phối Researcher, Analyst, Writer.

## Quy tắc quan trọng

- Không thêm agent nếu không có lý do rõ ràng.
- Mỗi agent phải có responsibility riêng.
- Shared state phải đủ rõ để debug.
- Phải có trace hoặc log cho từng bước.
- Phải benchmark, không chỉ nhìn output bằng cảm tính.

## Milestone 1: Baseline

File gợi ý:

- `src/multi_agent_research_lab/cli.py`
- `src/multi_agent_research_lab/services/llm_client.py`

TODO(student): thay baseline placeholder bằng một call LLM thật.

## Milestone 2: Supervisor

File gợi ý:

- `src/multi_agent_research_lab/agents/supervisor.py`
- `src/multi_agent_research_lab/graph/workflow.py`

TODO(student): implement routing policy.

Gợi ý câu hỏi thiết kế:

- Khi nào gọi Researcher?
- Khi nào gọi Analyst?
- Khi nào gọi Writer?
- Khi nào stop?
- Nếu agent fail thì retry hay fallback?

## Milestone 3: Worker agents

File gợi ý:

- `src/multi_agent_research_lab/agents/researcher.py`
- `src/multi_agent_research_lab/agents/analyst.py`
- `src/multi_agent_research_lab/agents/writer.py`

TODO(student): implement từng worker.

## Milestone 4: Trace và benchmark

File gợi ý:

- `src/multi_agent_research_lab/observability/tracing.py`
- `src/multi_agent_research_lab/evaluation/benchmark.py`
- `src/multi_agent_research_lab/evaluation/report.py`

Benchmark tối thiểu:

| Metric | Cách đo gợi ý |
|---|---|
| Latency | wall-clock time |
| Cost | token usage hoặc provider usage |
| Quality | rubric 0-10 do peer review |
| Citation coverage | số claims có source / tổng claims chính |
| Failure rate | số query fail / tổng query |

## Troubleshooting

### macOS: lỗi SSL certificate khi gọi API qua HTTPS (Tavily, OpenAI, ...)

Triệu chứng: khi implement `SearchClient` (hoặc bất kỳ HTTPS call nào) trên macOS, bạn có thể gặp lỗi kiểu:

```
ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed:
unable to get local issuer certificate
```

Nguyên nhân: Python cài từ python.org trên macOS **không dùng** certificate store của hệ điều hành, nên không tìm thấy CA bundle hợp lệ. Đây là lỗi môi trường, **không phải** do API key sai.

Cách khắc phục (chọn 1 trong 3):

1. **Chạy script cài certificate đi kèm Python** (nhanh nhất):

   ```bash
   /Applications/Python\ 3.12/Install\ Certificates.command
   ```

   (thay `3.12` bằng version Python của bạn)

2. **Dùng `certifi` trong code** — thêm `certifi` vào dependencies, rồi tạo SSL context khi gọi HTTPS:

   ```python
   import certifi
   import ssl
   from urllib.request import urlopen

   ssl_context = ssl.create_default_context(cafile=certifi.where())
   urlopen(request, timeout=timeout, context=ssl_context)
   ```

3. **Set biến môi trường** trỏ tới CA bundle của certifi (không cần đổi code):

   ```bash
   export SSL_CERT_FILE=$(python -m certifi)
   ```

## Exit ticket

Mỗi nhóm trả lời 2 câu:

1. Case nào nên dùng multi-agent? Vì sao?
2. Case nào không nên dùng multi-agent? Vì sao?

---

### Câu trả lời (2A202601310 - Trần Văn Đông)

**1. Case nào nên dùng multi-agent? Vì sao?**

Nên dùng multi-agent khi bài toán có **nhiều giai đoạn tách biệt rõ ràng** và mỗi giai đoạn
cần sự tập trung, kỹ năng khác nhau:

- **Research report / Due diligence**: Researcher tìm nguồn, Analyst đánh giá chất lượng nguồn
  và xác định claim chính, Writer tổng hợp bài viết có trích dẫn. Mỗi agent chỉ làm một việc,
  tránh "context pollution" khi nhồi tất cả vào một prompt.
- **Khi cần audit trail**: Biết agent nào làm gì, khi nào, với input/output gì — phục vụ
  compliance, explainability, debugging.
- **Khi quality quan trọng hơn latency**: Benchmark cho thấy multi-agent đạt 8.0/10 vs 5.0/10,
  với citation coverage 80% vs 40% so với single-agent.
- **Khi task dài và phức tạp**: Nhiều sub-task có thể cần retry độc lập (nếu Researcher fail,
  chỉ cần retry Researcher, không cần chạy lại từ đầu).

**2. Case nào không nên dùng multi-agent? Vì sao?**

Không nên dùng khi:

- **Latency là yếu tố quyết định**: Chat real-time cần < 2s — multi-agent tốn ~4.8× thời gian
  (19s vs 4s) do nhiều LLM call nối tiếp nhau.
- **Task đơn giản**: Câu hỏi tra cứu ("Paris là thủ đô của nước nào?"), tóm tắt văn bản ngắn,
  phân loại label — một LLM call là đủ, thêm Supervisor/Researcher/Analyst chỉ tốn tài nguyên
  mà không cải thiện chất lượng.
- **Chi phí token cần tối thiểu**: Multi-agent tốn ~4.2× token so với single-agent vì 3-4 LLM
  calls thay vì 1. Với scale lớn (triệu query/ngày), điều này có thể không tối ưu.
- **Không có ranh giới task tự nhiên**: Nếu buộc phải chia vai mà không có sự tách biệt rõ ràng,
  overhead handoff > benefit chuyên môn hoá.
- **Prototype nhanh / MVP**: Complexity của multi-agent (shared state, routing, guardrails)
  làm chậm tốc độ phát triển ban đầu.
