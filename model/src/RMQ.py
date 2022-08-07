import pika


class BasicRMQClient:
    """A basic RabbitMQ client handling connection failures"""

    def __init__(self, server, port, user, password, virtual_host='/'):
        self.channel = None
        self.connection = None
        self.server = server
        self.port = port
        self.user = user
        self.password = password
        self.virtual_host = virtual_host

    # publishes a message to the specified exchange on the active channel
    @staticmethod
    def publish_exchange(channel, exchange, body, routing_key=''):
        channel.basic_publish(exchange=exchange, body=body, routing_key=routing_key)

    # processes message from RabbitMQ
    def process(self, callback_on_message, source_queue):
        # define our connection parameters
        creds = pika.PlainCredentials(self.user, self.password)
        connection_params = pika.ConnectionParameters(host=self.server,
                                                      port=self.port,
                                                      virtual_host=self.virtual_host,
                                                      credentials=creds)
        # Connect to RMQ and wait until a message is received
        while True:
            try:
                print("Connecting to %s" % self.server)
                self.connection = pika.BlockingConnection(connection_params)

                # create channel and a queue bound to the source exchange
                self.channel = self.connection.channel()
                self.channel.basic_qos(prefetch_count=1)
                self.channel.basic_consume(
                        queue=source_queue, on_message_callback=callback_on_message, auto_ack=False)

                # print(' [*] Waiting for messages. To exit press CTRL+C')
                try:
                    self.channel.start_consuming()
                except KeyboardInterrupt:
                    self.channel.stop_consuming()
                    self.connection.close()
                    break
            # Recover from server-initiated connection closure - handles manual RMQ restarts
            except pika.exceptions.ConnectionClosedByBroker:
                continue
            # Do not recover on channel errors
            except pika.exceptions.AMQPChannelError as err:
                print("Channel error: {}, stopping...".format(err))
                break
            # Recover on all other connection errors
            except pika.exceptions.AMQPConnectionError:
                continue
