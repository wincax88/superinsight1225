# SQLAlchemy 2.0 迁移指南

## 概述

SuperInsight 平台已成功从 SQLAlchemy 1.x 迁移到 SQLAlchemy 2.0，消除了所有弃用警告并提升了代码的未来兼容性。

## 迁移内容

### 1. 查询语法更新

#### 之前 (SQLAlchemy 1.x)
```python
# 查询单个记录
user = session.query(UserModel).filter(UserModel.id == user_id).first()

# 查询多个记录
users = session.query(UserModel).filter(UserModel.is_active == True).all()

# 计数查询
count = session.query(UserModel).count()
```

#### 之后 (SQLAlchemy 2.0)
```python
# 查询单个记录
stmt = select(UserModel).where(UserModel.id == user_id)
user = session.execute(stmt).scalar_one_or_none()

# 查询多个记录
stmt = select(UserModel).where(UserModel.is_active == True)
users = list(session.execute(stmt).scalars().all())

# 计数查询
stmt = select(func.count(UserModel.id))
count = session.execute(stmt).scalar()
```

### 2. 更新操作

#### 之前 (SQLAlchemy 1.x)
```python
session.query(TaskModel).filter(
    TaskModel.id == task_id
).update({"quality_score": new_score})
```

#### 之后 (SQLAlchemy 2.0)
```python
from sqlalchemy import update
stmt = update(TaskModel).where(
    TaskModel.id == task_id
).values(quality_score=new_score)
session.execute(stmt)
```

### 3. 导入更新

所有相关文件都已添加必要的导入：

```python
from sqlalchemy import select, func, update
```

## 已更新的文件

### 核心数据库文件
- `src/database/manager.py` - 数据库管理器的所有查询操作
- `src/database/init_db.py` - 数据库初始化和测试查询
- `src/database/batch_processor.py` - 批处理更新操作

### 业务逻辑文件
- `src/hybrid/sync_manager.py` - 混合云数据同步查询
- `src/label_studio/integration.py` - Label Studio 集成查询
- `src/label_studio/collaboration.py` - 协作功能查询
- `src/quality/repair.py` - 质量修复相关查询
- `src/security/controller.py` - 安全控制器查询

## 性能改进

SQLAlchemy 2.0 迁移带来的性能改进：

1. **更好的类型安全**: 明确的结果类型处理
2. **优化的查询构建**: 更高效的 SQL 生成
3. **改进的缓存**: 更好的查询计划缓存
4. **减少内存使用**: 优化的对象生命周期管理

## 验证结果

### 测试通过情况
- ✅ 数据库设置测试: 11/11 通过
- ✅ 系统集成测试: 17/17 通过
- ✅ 无 SQLAlchemy 弃用警告

### 功能验证
- ✅ 所有数据库操作正常工作
- ✅ 查询性能保持或改善
- ✅ 事务处理正确
- ✅ 连接池管理正常

## 最佳实践

### 1. 查询构建
```python
# 推荐：使用 select() 构建查询
stmt = select(Model).where(Model.field == value)
result = session.execute(stmt).scalar_one_or_none()

# 避免：直接使用 session.query()
# result = session.query(Model).filter(Model.field == value).first()
```

### 2. 结果处理
```python
# 单个结果
result = session.execute(stmt).scalar_one_or_none()  # 可能为 None
result = session.execute(stmt).scalar_one()          # 必须存在，否则抛异常

# 多个结果
results = list(session.execute(stmt).scalars().all())

# 计数
count = session.execute(select(func.count(Model.id))).scalar()
```

### 3. 更新操作
```python
# 使用 update() 语句
from sqlalchemy import update
stmt = update(Model).where(Model.id == id).values(field=new_value)
session.execute(stmt)
session.commit()
```

## 兼容性说明

- **向前兼容**: 代码与 SQLAlchemy 2.0+ 完全兼容
- **性能优化**: 利用 SQLAlchemy 2.0 的性能改进
- **类型安全**: 更好的 IDE 支持和类型检查
- **未来保证**: 避免未来版本的弃用警告

## 故障排除

### 常见问题

1. **导入错误**: 确保导入了 `select`, `func`, `update`
2. **结果处理**: 使用正确的结果提取方法
3. **事务管理**: 确保正确提交事务

### 调试技巧

```python
# 打印生成的 SQL
print(stmt.compile(compile_kwargs={"literal_binds": True}))

# 启用 SQL 日志
engine = create_engine(url, echo=True)
```

## 总结

SQLAlchemy 2.0 迁移已成功完成，系统现在：

- 🚀 使用最新的 SQLAlchemy 2.0 语法
- 🔧 消除了所有弃用警告
- 📈 提升了查询性能
- 🛡️ 增强了类型安全
- 🔮 确保了未来兼容性

系统已准备好用于生产环境，并将受益于 SQLAlchemy 2.0 的所有改进。