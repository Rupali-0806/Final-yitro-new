import ContactCard from './ContactCard';

interface ContactsGridProps {
  onContactClick: (id: number) => void;
}

export default function ContactsGrid({ onContactClick }: ContactsGridProps) {
  const contacts = [
    { id: 1, name: 'Contact - 1' },
    { id: 2, name: 'Contact - 1' },
    { id: 3, name: 'Contact - 1' },
    { id: 4, name: 'Contact - 1' }
  ];

  return (
    <div className="p-6">
      <div className="max-w-4xl mx-auto">
        <div className="grid grid-cols-2 gap-8 max-w-2xl mx-auto">
          {contacts.map((contact) => (
            <ContactCard
              key={contact.id}
              id={contact.id}
              name={contact.name}
              onClick={onContactClick}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
