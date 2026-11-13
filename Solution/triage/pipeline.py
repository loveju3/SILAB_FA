"""Chronological inference with state scoped to one input file."""
from time import perf_counter

from .classification import CategoryClassifier
from .contracts import Prediction, Ticket, validate_predictions
from .devices import DeviceResolver
from .rules import resolve_context, suggestion, urgency
from .text import classification_text, urgency_text


def predict(tickets: list[Ticket], resolver: DeviceResolver,
            classifier: CategoryClassifier) -> tuple[list[Prediction], list[dict]]:
    previous, duplicates, outputs, timings = {}, {}, {}, []
    ordered_tickets = sorted(tickets, key=lambda t: t.created_at)
    batch = None
    batch_texts = None
    classify_many = getattr(classifier, "classify_many", None)
    if callable(classify_many):
        # The frozen encoder handles each text independently. Computing features
        # ahead of time must not read reporter context or gold/future answers.
        batch_texts = [classification_text(t.raw_text) for t in ordered_tickets]
        if classifier.mask_device_names:
            batch_texts = [resolver.mask_names(text) for text in batch_texts]
        batch = classify_many(batch_texts)
        if len(batch.predictions) != len(tickets) or len(batch.batch_seconds) != len(tickets):
            raise ValueError("批次分類結果筆數不一致")
    # Python's stable sort preserves input order for identical timestamps.
    for index, ticket in enumerate(ordered_tickets):
        start = perf_counter()
        text = classification_text(ticket.raw_text)
        candidates = resolver.candidates(text)
        if batch is not None:
            category = batch.predictions[index]
        else:
            features = resolver.mask_names(text) if classifier.mask_device_names else text
            category = classifier.classify(features)
        context, category = resolve_context(ticket, text, candidates, category,
                                             previous.get(ticket.reporter))
        critical = bool(context.device_id and resolver.devices[context.device_id]["critical"])
        level = urgency(urgency_text(ticket.raw_text), critical)
        key = (ticket.reporter, context.device_id, category.category)
        duplicate = duplicates.get(key) if context.device_id and category.category else None
        if context.device_id and category.category:
            duplicates[key] = ticket.ticket_id
        outputs[ticket.ticket_id] = Prediction(ticket.ticket_id, context.device_id, category.category,
                                               level, duplicate, False,
                                               suggestion(ticket, category, level))
        previous[ticket.reporter] = context
        duration = perf_counter() - start
        batch_duration = batch.batch_seconds[index] if batch is not None else 0.0
        timings.append({"ticket_id": ticket.ticket_id, "seconds": duration + batch_duration,
                        "batch_classification_seconds": batch_duration})
    ordered = [outputs[t.ticket_id] for t in tickets]
    validate_predictions(tickets, ordered, set(resolver.devices))
    return ordered, timings
