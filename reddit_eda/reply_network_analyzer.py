import json


class ReplyNetworkAnalyzer:

    ACTIVITY_COLUMNS = [
        "subreddit",
        "month",
        "user",
        "replies_sent",
        "unique_users_replied_to",
        "replies_received",
        "unique_users_who_replied",
        "total_interactions"
    ]

    RETURN_COLUMNS = [
        "subreddit",
        "month",
        "user",
        "returned_next_month"
    ]

    COMMUNITY_COLUMNS = [
        "subreddit",
        "month",
        "active_users",
        "edges",
        "density",
        "average_out_degree",
        "average_in_degree"
    ]

    RECIPROCITY_COLUMNS = [
        "subreddit",
        "month",
        "user",
        "reciprocal_partners",
        "outgoing_partners",
        "proportional_reciprocity"
    ]

    def __init__(self, file_path, subreddit_name):
        self.file_path = file_path
        self.subreddit_name = subreddit_name

        # one dict per month: user -> list of users they replied to
        with open(file_path, "r") as f:
            self.months = json.load(f)


    @staticmethod
    def incoming_users(month_data):

        incoming = {}

        for user, replied_to in month_data.items():

            for target in replied_to:

                if target not in incoming:
                    incoming[target] = set()

                incoming[target].add(user)

        return incoming


    def user_activity_rows(self):

        rows = []

        for month_index, month_data in enumerate(self.months):

            incoming_counts = {}

            for user, replied_to in month_data.items():

                for target in replied_to:

                    incoming_counts[target] = (
                        incoming_counts.get(target, 0) + 1
                    )

            incoming = self.incoming_users(month_data)

            for user, replied_to in month_data.items():

                replies_sent = len(replied_to)
                replies_received = incoming_counts.get(user, 0)

                rows.append({
                    "subreddit": self.subreddit_name,
                    "month": month_index + 1,
                    "user": user,
                    "replies_sent": replies_sent,
                    "unique_users_replied_to": len(set(replied_to)),
                    "replies_received": replies_received,
                    "unique_users_who_replied":
                        len(incoming.get(user, set())),
                    "total_interactions":
                        replies_sent + replies_received
                })

        return rows


    def user_return_rows(self):

        rows = []

        # the last month has no next month to return in
        for t in range(len(self.months) - 1):

            next_users = set(self.months[t + 1].keys())

            for user in self.months[t]:

                rows.append({
                    "subreddit": self.subreddit_name,
                    "month": t + 1,
                    "user": user,
                    "returned_next_month": int(user in next_users)
                })

        return rows


    def community_rows(self):

        rows = []

        for month_index, month_data in enumerate(self.months):

            # directed reply network: one edge per (user, target) pair
            nodes = set()
            number_of_edges = 0

            for user, replied_to in month_data.items():

                targets = set(replied_to)

                if targets:
                    nodes.add(user)
                    nodes.update(targets)
                    number_of_edges += len(targets)

            number_of_users = len(nodes)

            if number_of_users > 1 and number_of_edges > 0:
                density = (
                    number_of_edges
                    / (number_of_users * (number_of_users - 1))
                )
            else:
                density = 0.0

            if number_of_users > 0:
                average_degree = number_of_edges / number_of_users
            else:
                average_degree = 0.0

            # every edge has one sender and one receiver,
            # so average in-degree equals average out-degree
            rows.append({
                "subreddit": self.subreddit_name,
                "month": month_index + 1,
                "active_users": number_of_users,
                "edges": number_of_edges,
                "density": density,
                "average_out_degree": average_degree,
                "average_in_degree": average_degree
            })

        return rows


    def reciprocity_rows(self):

        rows = []

        for month_index, month_data in enumerate(self.months):

            incoming = self.incoming_users(month_data)

            for user, replied_to in month_data.items():

                outgoing_users = set(replied_to)
                outgoing_count = len(outgoing_users)

                reciprocal_count = len(
                    outgoing_users
                    .intersection(incoming.get(user, set()))
                )

                if outgoing_count > 0:
                    proportional_reciprocity = (
                        reciprocal_count / outgoing_count
                    )
                else:
                    proportional_reciprocity = 0.0

                rows.append({
                    "subreddit": self.subreddit_name,
                    "month": month_index + 1,
                    "user": user,
                    "reciprocal_partners": reciprocal_count,
                    "outgoing_partners": outgoing_count,
                    "proportional_reciprocity":
                        proportional_reciprocity
                })

        return rows
