from dataclasses import dataclass

import randomname

from models.llm import LLM
from models.configs import create_model_flags
from models.utils import set_seed
from training.params import TrainingArgs, DataArgs, TeacherArgs, StudentArgs
from training.trainer import Trainer


@dataclass
class AllArgs(TrainingArgs, DataArgs, TeacherArgs, StudentArgs):
    def __post_init__(self):
        StudentArgs.__post_init__(self)
        TrainingArgs.__post_init__(self)


def main(args: AllArgs):
    if args.seed:
        set_seed(args.seed)

    if not args.run_name:
        args.run_name = randomname.get_name()
    if not args.group_name:
        args.group_name = "initial_runs"

    if args.distractor_dataset and "llama3" not in args.base:
        raise NotImplementedError("Distractors only tested for LLama3 models. If you want to use distractors for Qwen, comment out this line and make sure that the outputs are reasonable.")

    base_llm = LLM(args.base, opening_message=args.opening_message)
    if args.chat_template_path:
        base_llm.set_chat_template(args.chat_template_path)
    if args.knowledge_cutoff != "default":
        base_llm.set_knowledge_cutoff(args.knowledge_cutoff)

    # Create teacher LLM for tokenization if using a separate teacher model
    # NOTE: teacher model loading is deferred to Trainer._init_models() so that
    # accelerator is initialized first and DeepSpeed can shard the teacher properly.
    teacher_llm = None
    if args.logit_loss_weight and args.teacher not in {"student", "student_base"}:
        from models.configs import MODEL_CONFIGS
        from models.messages import Message, Role
        teacher_config = MODEL_CONFIGS[args.teacher]
        teacher_opening = Message(Role.SYSTEM, teacher_config.system_message)
        teacher_llm = LLM(args.teacher, opening_message=teacher_opening)
    # Build data file list
    if args.custom_train_data:
        train_file = args.custom_train_data
        val_file = args.custom_val_data if args.custom_val_data else None
        args.datapath = ""  # custom paths are absolute, not relative to datapath
    else:
        from curriculum.naming import generate_lesson_name, generate_exam_name, generate_augmented_filename
        lesson_model_flags = create_model_flags(args.lesson_model)
        exam_model_flags = create_model_flags(args.exam_model)

        train_file = generate_augmented_filename(
            lesson_filename=generate_lesson_name(
                dataset_family=args.dataset_family, dataset=args.dataset,
                variant=args.variant, model=args.question_model,
                questions=args.train_questions, temperature=args.question_temperature,
                max_items=args.max_items_train,
            ),
            n_choices=args.lesson_num_choices, temperature=args.lesson_temp,
            model_flags=lesson_model_flags,
            partition_idx=args.partition_idx, partition_type=args.partition_type,
        )

        val_file = generate_augmented_filename(
            lesson_filename=generate_exam_name(
                dataset_family=args.dataset_family, dataset=args.dataset,
                variant=args.variant, max_items=args.max_items_test,
            ),
            n_choices=args.exam_num_choices, temperature=args.exam_temp,
            model_flags=exam_model_flags,
        )

    data = [[train_file], [val_file] if (args.validate and val_file) else []]

    # Load tools schema if use_tool_token is enabled
    tools = None
    if args.use_tool_token and args.tools_schema_path:
        import json
        with open(args.tools_schema_path) as f:
            tools = json.load(f)
        print(f"Loaded {len(tools)} tool definitions from {args.tools_schema_path}", flush=True)

    trainer = Trainer(base_llm=base_llm, data=data, hp=args, teacher_llm=teacher_llm, tools=tools)
    trainer.train()


if __name__ == "__main__":
    from jsonargparse import CLI
    main(CLI(AllArgs, as_positional=False))
