
---

## `nn.ModuleList` — 注册子模块的列表

```python
# ❌ 普通 list：不会被 PyTorch 注册为子模块
self.blocks = [Block(config) for _ in range(n)]  # 参数不被 .to(device) 管理

# ✅ ModuleList：自动注册所有子模块
self.blocks = nn.ModuleList([Block(config) for _ in range(n)])
```

**普通 list vs `nn.ModuleList`：**

| | 普通 list | `nn.ModuleList` |
|---|---|---|
| 参数被 `.to(device)` 管理 | ❌ | ✅ |
| 被 `state_dict()` 包含 | ❌ | ✅ |
| 在 `model.parameters()` 中出现 | ❌ | ✅ |
| 遍历/索引 | ✅ | ✅ |

`nn.ModuleDict` 同理——当需要用 dict 管理多个 `nn.Module` 时，用 `nn.ModuleDict` 而不是普通 dict。
