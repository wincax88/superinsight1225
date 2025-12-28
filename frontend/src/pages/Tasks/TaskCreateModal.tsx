// Task creation modal component
import { Modal, Form, Input, Select, DatePicker, message } from 'antd';
import { useCreateTask } from '@/hooks/useTask';
import type { CreateTaskPayload, TaskPriority, AnnotationType } from '@/types';

const { TextArea } = Input;

interface TaskCreateModalProps {
  open: boolean;
  onCancel: () => void;
  onSuccess: () => void;
}

const priorityOptions: { label: string; value: TaskPriority }[] = [
  { label: 'Low', value: 'low' },
  { label: 'Medium', value: 'medium' },
  { label: 'High', value: 'high' },
  { label: 'Urgent', value: 'urgent' },
];

const annotationTypeOptions: { label: string; value: AnnotationType }[] = [
  { label: 'Text Classification', value: 'text_classification' },
  { label: 'Named Entity Recognition (NER)', value: 'ner' },
  { label: 'Sentiment Analysis', value: 'sentiment' },
  { label: 'Question & Answer', value: 'qa' },
  { label: 'Custom', value: 'custom' },
];

export const TaskCreateModal: React.FC<TaskCreateModalProps> = ({
  open,
  onCancel,
  onSuccess,
}) => {
  const [form] = Form.useForm();
  const createTask = useCreateTask();

  const handleSubmit = async () => {
    try {
      const values = await form.validateFields();
      const payload: CreateTaskPayload = {
        name: values.name,
        description: values.description,
        priority: values.priority,
        annotation_type: values.annotation_type,
        due_date: values.due_date?.toISOString(),
        tags: values.tags,
      };

      await createTask.mutateAsync(payload);
      form.resetFields();
      onSuccess();
    } catch (error) {
      if (error instanceof Error && error.message !== 'Validation failed') {
        message.error('Failed to create task');
      }
    }
  };

  return (
    <Modal
      title="Create New Task"
      open={open}
      onCancel={onCancel}
      onOk={handleSubmit}
      confirmLoading={createTask.isPending}
      width={600}
      destroyOnClose
    >
      <Form
        form={form}
        layout="vertical"
        initialValues={{
          priority: 'medium',
          annotation_type: 'text_classification',
        }}
      >
        <Form.Item
          name="name"
          label="Task Name"
          rules={[
            { required: true, message: 'Please enter task name' },
            { max: 100, message: 'Task name cannot exceed 100 characters' },
          ]}
        >
          <Input placeholder="Enter task name" />
        </Form.Item>

        <Form.Item
          name="description"
          label="Description"
          rules={[{ max: 500, message: 'Description cannot exceed 500 characters' }]}
        >
          <TextArea rows={3} placeholder="Enter task description (optional)" />
        </Form.Item>

        <Form.Item
          name="annotation_type"
          label="Annotation Type"
          rules={[{ required: true, message: 'Please select annotation type' }]}
        >
          <Select options={annotationTypeOptions} placeholder="Select annotation type" />
        </Form.Item>

        <Form.Item
          name="priority"
          label="Priority"
          rules={[{ required: true, message: 'Please select priority' }]}
        >
          <Select options={priorityOptions} placeholder="Select priority" />
        </Form.Item>

        <Form.Item name="due_date" label="Due Date">
          <DatePicker style={{ width: '100%' }} placeholder="Select due date (optional)" />
        </Form.Item>

        <Form.Item name="tags" label="Tags">
          <Select
            mode="tags"
            placeholder="Add tags (press Enter to add)"
            tokenSeparators={[',']}
          />
        </Form.Item>
      </Form>
    </Modal>
  );
};
