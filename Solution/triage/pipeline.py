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
    # Python's stable sort preserves input order for identical timestamps.
    for ticket in sorted(tickets, key=lambda t: t.created_at):
        start = perf_counter()
        text = classification_text(ticket.raw_text)
        candidates = resolver.candidates(text)
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
        timings.append({"ticket_id": ticket.ticket_id, "seconds": perf_counter() - start})
    ordered = [outputs[t.ticket_id] for t in tickets]
    validate_predictions(tickets, ordered, set(resolver.devices))
    return ordered, timings
